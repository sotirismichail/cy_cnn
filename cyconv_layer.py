import cyconv

# todo: implement backward and forward propagation convolution code

class cyconv_layer:
    @staticmethod
    def forward(ctx,
                matrix,
                weight,
                workspace,
                stride: int = 1,
                padding: int = 0,
                dilation: int = 1):
        ctx.matrix = matrix
        ctx.weight = weight
        ctx.workspace = workspace
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        output = cyconv.forward(matrix, weight, workspace, stride, padding, dilation)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = cyconv.backward()