-- From Yoon's char lstm code
local HighwayMLP = {}

function HighwayMLP.mlp(size, num_layers, dropout,bias, f)
  -- size = dimensionality of inputs
  -- num_layers = number of hidden layers (default = 1)
  -- bias = bias for transform gate (default = -2)
  -- f = non-linearity (default = ReLU)
  -- dropout: dropout probability between two layers

  local output, transform_gate, carry_gate
  local num_layers = num_layers or 1
  local bias = bias or -2
  local f = f or nn.ReLU()
  local input = nn.Identity()()
  local inputs = {[1]=input}
  for i = 1, num_layers do
    local linear = nn.Linear(size, size)      
    output = f(linear(inputs[i]))
    linear.bias:uniform(-0.01, 0.01)
    linear.weight:uniform(-0.01, 0.01)
    local transform_linear = nn.Linear(size, size)
    transform_linear.bias:uniform(-0.01, 0.01)
    transform_linear.weight:uniform(-0.01, 0.01)
    transform_gate = nn.Sigmoid()(nn.AddConstant(bias)(transform_linear(inputs[i])))
    carry_gate = nn.AddConstant(1)(nn.MulConstant(-1)(transform_gate))
    output = nn.CAddTable()({
      nn.CMulTable()({transform_gate, output}),
      nn.CMulTable()({carry_gate, inputs[i]})  })
    
    if i == num_layers then
		table.insert(inputs, output)
	else
		local dropped_out = nn.Dropout(dropout)(output)
		table.insert(inputs, dropped_out)
	end
  end

  return nn.gModule({input},{output})
end

return HighwayMLP
