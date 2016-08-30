require 'torch'
require 'nn'
require 'nngraph'
require 'xlua'
require 'gnuplot'
require 'image'
require 'optim'

params     = {}
config     = {}
train_data = {}
test_data  = {}
net        = {}
criterion  = {}

function Parse()
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options for MNIST')
  cmd:text()

  cmd:option('-data_path', 'data', 'Path to data folder')
  cmd:option('-train_images', 'train.img', 'Training images file')
  cmd:option('-train_labels', 'train.out', 'Training labels file')
  cmd:option('-test_images', 'test.img', 'Test images file')
  cmd:option('-test_labels', 'test.out', 'Test labels file')
  cmd:option('-gradcheck', false, 'Gradident check')
  cmd:option('-num_epochs', 10, 'Number of epochs)')
  local config = cmd:parse(arg)
  return config
end

-- expecting img as a 2-D tensor with size w x h
function ShowImages(dataset, num_img)
  local num_rows = math.floor(math.sqrt(num_img))
  local num_cols = math.floor(num_img / num_rows)
  assert(num_rows * num_cols == num_img, 'num_img must be a perfect square')
  local w,h = dataset.width, dataset.height

  local tensor = torch.Tensor(w*num_rows, h*num_cols):zero()
  local cur_row = 1
  local cur_col = 1
  local img_idx = torch.Tensor(num_img):uniform(0, 1):mul(dataset.num_img * 0.4):floor():add(1)

  for i = 1,num_img do
    local cur_img = img_idx[i]
    local img = torch.Tensor(w,h):copy(dataset.img[cur_img])
    tensor:sub(w*(cur_row-1)+1,w*cur_row,h*(cur_col-1)+1,h*cur_col):copy(img:t())

    cur_col = cur_col + 1
    if cur_col > num_cols then
      cur_col = 1
      cur_row = cur_row + 1
    end
  end

  gnuplot.epsfigure('sample.eps')
  gnuplot.imagesc(image.rotate(tensor, math.pi/2), 'gray')
  gnuplot.title(string.format('%d sample MNIST digits', num_img))
  gnuplot.raw('set key font ",30"')
  gnuplot.raw('unset xtics; unset ytics; unset colorbox')

  gnuplot.plotflush()
end

function LoadData(images, labels)
  local function Bytes2Num(bytes)
    local ans = 0
    for i = 1,4 do
      ans = ans*256 + bytes[i]
    end
    return ans
  end

  print(string.format('# Loading data from file \'%s\' and \'%s\'', images, labels))

  -- read images
  local img_file = assert(io.open(images, 'rb'), string.format('Cannot open file %s', images))
  local dummy   = Bytes2Num({img_file:read(4):byte(1,4)})
  local num_img = Bytes2Num({img_file:read(4):byte(1,4)})
  local height  = Bytes2Num({img_file:read(4):byte(1,4)})
  local width   = Bytes2Num({img_file:read(4):byte(1,4)})
  local img = torch.FloatTensor(num_img, width*height)
  for i = 1,num_img do
    local raw_img = img_file:read(width*height)
    if raw_img == nil then break end
    local cur_img = {raw_img:byte(1, width*height)}
    img[i]:copy(torch.FloatTensor(cur_img))
    xlua.progress(i, num_img)
  end
  img_file:close()
  print('  Loaded images')

  -- read labels
  local out_file = assert(io.open(labels, 'rb'), string.format('Cannot open file %s', labels))
  dummy   = Bytes2Num({out_file:read(4):byte(1,4)})
  num_img = Bytes2Num({out_file:read(4):byte(1,4)})
  local out     = torch.FloatTensor(num_img)
  for i = 1,num_img do
    local raw_img = out_file:read(1)
    if raw_img == nil then break end
    out[i] = raw_img:byte(1,1)
    xlua.progress(i, num_img)
  end
  out_file:close()
  print('  Loaded labels')

  local dataset = {}
  dataset.num_img = num_img
  dataset.height  = height
  dataset.width   = width
  dataset.img     = img:div(255):add(-0.5)
  dataset.out     = out

  return dataset
end

function BuildConvNet1() 
  net = nn.Sequential()
  net:add(nn.Reshape(1, 28, 28))

  --layer 1
  net:add(nn.SpatialConvolution(1, 32, 5, 5, 1, 1, 2, 2))
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --layer 2
  net:add(nn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --layer 3
  net:add(nn.Reshape(7*7*64))
  net:add(nn.Linear(7*7*64, 1024))
  net:add(nn.ReLU())
  net:add(nn.Dropout(0.8))
  net:add(nn.Linear(1024, 10))
  net:add(nn.LogSoftMax())

  criterion = nn.ClassNLLCriterion()

  params.W, params.gradW = net:getParameters()
  if config.gradcheck then
    params.W:uniform(-1.0, 1.0)
  else
    params.W:uniform(-0.1, 0.1)
  end
  params.gradW:zero()
end

function prob_nn(img)
  local num = net:forward(img):exp()
  local den = num:sum()
  return num:div(den)
end

function f(from, to, dataset, num_trained)
  local W     = params.W
  local width, height = dataset.width, dataset.height
  local img = torch.Tensor(to-from+1,width*height):copy(dataset.img:sub(from,to,1,width*height)) 
  local out = torch.Tensor(to-from+1):copy(dataset.out:sub(from, to)):add(1)

  local net_out = net:forward(img)
  local loss    = criterion:forward(net_out, out)

  local d_net_out = criterion:backward(net_out, out)
  net:backward(img, d_net_out)

  if not config.gradcheck then xlua.progress(num_trained, dataset.num_img) end
  
  return loss, params.gradW
end

function Classify(img)
  local W = params.W
  local label = nil

  local maxProb = 0
  local p = prob_nn(img)
  for i = 1, 10 do
    if p[i] > maxProb then 
      maxProb = p[i]
      label = i;
    end
  end
  return label
end

function Test(dataset)
  net:evaluate()

  local num_correct = 0
  local num_img = dataset.num_img

  for i = 1,num_img do
    local img = dataset.img[i]
    local out = dataset.out[i] + 1

    local pred = Classify(img)
    if pred == out then
      num_correct = num_correct + 1
    end
  end

  net:training()
  return num_correct, num_img
end

function Train(train_data, test_data)
  print('# Start training')
  local learning_rate = 1e-4
  local lambda        = 0.005
  local num_img       = train_data.num_img
  local batch_size    = 50
  local prev_correct  = 0
  
  for epoch = 1,20 do
    if (epoch >= 15) then learning_rate = learning_rate * 0.8 end
    local loss = 0

    local max_size = math.ceil(num_img / batch_size)
    local num_trained = 0
    for i = 1, max_size do
      local from = (torch.random() % max_size + max_size) % max_size + 1
      from = (from-1)*batch_size + 1
      local to = from + batch_size

      if from > num_img then break end
      if to > num_img then to = num_img end

      --local l, gradW = fnn(from, to, train_data, lambda)
      num_trained = num_trained + batch_size
      local l, gradW = f(from, to, train_data, num_trained)
      loss = loss + l

      params.W:add(-learning_rate/batch_size, gradW)
    end

    local num_correct, num_img = Test(test_data)
    print(string.format('  Epoch: %d\n  Correct: %d / %d', epoch, num_correct, num_img))

    if num_correct < prev_correct then
      print(string.format('  learning_rate: %.10f -> %.10f', learning_rate, learning_rate*0.5))
      learning_rate = learning_rate * 0.5
    end
    prev_correct = num_correct
  end
  torch.save('woseed.t7', net)
  print('# Done')
end

function Main()
  torch.manualSeed(21260063)
  torch.setdefaulttensortype('torch.FloatTensor')
  config = Parse()
  print('Config')
  print(config)

  test_data  = LoadData(config.data_path..'/'..config.test_images,  config.data_path..'/'..config.test_labels)
  --BuildNet()
  BuildConvNet1()

  if config.gradcheck then
    train_data.num_img = 5
    GradCheck(test_data)
  else
    train_data = LoadData(config.data_path..'/'..config.train_images, config.data_path..'/'..config.train_labels)
    ShowImages(train_data, 64)

    local num_correct, num_img = Test(test_data)
    print()
    print(string.format('  Correct: %d / %d', num_correct, num_img))
    Train(train_data, test_data)
  end
end

Main()
