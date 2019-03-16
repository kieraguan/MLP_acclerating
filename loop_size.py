import time
import numpy as np

from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

def transpose_parallel(a_cpu):
    tile_x = np.int(32)  # tile size x
    tile_y = np.int(32)  # when change this, remember to change it in kernel!        tile size y
    transpose_kernel_code = """
    __global__ void Transpose(float *a, float *b, const unsigned int X, const unsigned int Y)
    {
        int ty = blockIdx.y * blockDim.y + threadIdx.y;
        int tx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ty < X && tx < Y){
            b[ty * Y + tx] = a[tx * X + ty];                                //in transpose, b[i,j]=a[j,i]
        }
    }
    """
    prg_transpose = SourceModule(transpose_kernel_code)

    [Y, X] = a_cpu.shape

    block_x = np.ceil(float(Y) / tile_x).astype(np.int)  # block size x
    block_y = np.ceil(float(X) / tile_y).astype(np.int)  # block size y

    start = time.time()
    a_gpu = gpuarray.to_gpu(a_cpu)
    a_transpose_gpu = gpuarray.empty((X, Y), a_cpu.dtype)
    time1 = time.time()-start

    evt = prg_transpose.get_function('Transpose')

    start = cuda.Event()
    end = cuda.Event()

    start.record()
    evt(a_gpu, a_transpose_gpu, np.uint32(X), np.uint32(Y),
        block = (tile_x, tile_y, 1),
       grid = (block_x, block_y, 1))
    
    end.record()
    end.synchronize()
    time2 = start.time_till(end)*1e-3

    start = time.time()
    a_transpose = a_transpose_gpu.get()
    time3 = time.time()-start
    
    return a_transpose, time1+time2+time3
class MatrixMultiply:
    tile_x = np.int(32)  # tile size x
    tile_y = np.int(32)  # when change this, remember to change it in kernel!        tile size y
    matrix_mul_optimized1_kernel_code = """
    __global__ void Matrix_multiply_optimized1(float *a, float *b, float *c, 
        const unsigned int X, const unsigned int Y, const unsigned int Z)
    {
        int TILE_WIDTH = blockDim.y;

        __shared__ float ds_A[32][32];
        __shared__ float ds_B[32][32];

        int bx = blockIdx.x;  
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int Row = by * blockDim.y + ty;  
        int Col = bx * blockDim.x + tx;
        float Cvalue = 0;

        for(int t=0; t < ((X-1)/TILE_WIDTH + 1); ++t){
            if(Row < Y && t*TILE_WIDTH + tx < X){
                ds_A[ty][tx] = a[Row * X + t*TILE_WIDTH + tx];
            }
            else{
                ds_A[ty][tx] = 0.0;
            }
            if(Col < Z && t*TILE_WIDTH + ty < X){
                ds_B[ty][tx] = b[(t*TILE_WIDTH + ty)* Z + Col];
            }
            else{
                ds_B[ty][tx] = 0.0;
            }
            __syncthreads();

            for(int i = 0; i < TILE_WIDTH; ++i){
                Cvalue += ds_A[ty][i]*ds_B[i][tx];
            }
            __syncthreads();
        }
        if(Row < Y && Col < Z){
            c[Row*Z + Col] = Cvalue;
        }
    }
    """
    prg_matrix_mul_optimized1 = SourceModule(matrix_mul_optimized1_kernel_code)

    def prepare_date(self, a_cpu,b_cpu):
        a_cpu = np.float32(a_cpu)
        b_cpu = np.float32(b_cpu)
            # self.a_gpu = cl.array.to_device(MatrixMultiply.queue, a_cpu)  # send it to device
        start=time.time()
        self.a_gpu = gpuarray.to_gpu(a_cpu)
     
        self.b_gpu = gpuarray.to_gpu(b_cpu)
        self.time_=time.time()-start
       
    def matrix_mul_tile(self, a_cpu,b_cpu,tile_x,tile_y):
        self.prepare_date(a_cpu,b_cpu)
        (Y, X) = a_cpu.shape
        (X,Z)=b_cpu.shape

        block_x = np.ceil(float(Y) / tile_x).astype(np.int)  # block size x
        block_y = np.ceil(float(X) / tile_y).astype(np.int)  # block size y
        start=time.time()
        c_tile1_gpu = gpuarray.empty((Y, Z), a_cpu.dtype)
        time1=time.time()-start
        evt = MatrixMultiply.prg_matrix_mul_optimized1.get_function('Matrix_multiply_optimized1')

        start = cuda.Event()
        end = cuda.Event()

        start.record()
        evt(self.a_gpu, self.b_gpu, c_tile1_gpu, np.uint32(X), np.uint32(Y), np.uint32(Z),
            block = (tile_x, tile_y, 1),
            grid = (block_x, block_y, 1))
        # evt(self.a_gpu, self.b_gpu, c_naive_gpu, np.uint32(X), np.uint32(Y), np.uint32(Z),
        #     block = (tile_x,tile_y, 1),
        #     grid = (block_x, block_y, 1))
        end.record()
        end.synchronize()
        time_tile1= start.time_till(end)*1e-3
        start=time.time()
        c_tile1 = c_tile1_gpu.get()
        time2=time.time()-start
        #print("naive mul", time_naive,time1,time2,time_naive+time1+time2+self.time_ )
        return c_tile1, time_tile1+time1+time2+self.time_

  
    
def matmul_serial(a,b):
    y,x = a.shape
    x,z = b.shape
    c = np.zeros((y,z))
    for i in range(y):
        for j in range(z):
            for k in range(x):
                c[i][j] += a[i][k]*b[k][j]
    
    return c
    
def affine_forward_serial(x, w, b):
    
    N = x.shape[0]
    D = w.shape[0]
    X = x.reshape(N,D)
    out = matmul_serial(X,w) + b
    
    return out

def affine_backward_serial(dout, x, w, b):
    
    db = np.sum(dout, axis=0)
    N = x.shape[0]
    D = w.shape[0]
    X = x.reshape(N,D)
    dw = matmul_serial(X.T, dout)
    dx = matmul_serial(dout, w.T).reshape(x.shape)
    
    return dx, dw, db

def affine_forward_parallel_tile(x, w, b,tile_x,tile_y):
    mul = MatrixMultiply()
    start = time.time()
    N = x.shape[0]
    D = w.shape[0]
    X = x.reshape(N,D)
    time1 = time.time()-start
    out, time2 = mul.matrix_mul_tile(X,w,tile_x,tile_y)
    start = time.time()
    out = out + b
    time3 = time.time()-start
    
    return out, time1+time2+time3

def affine_backward_parallel_tile(dout, x, w, b,tile_x,tile_y):
    mul = MatrixMultiply()
    start = time.time()
    db = np.sum(dout, axis=0)
    N = x.shape[0]
    D = w.shape[0]
    X = x.reshape(N,D)
    time1 = time.time()-start
    XT,time2 = transpose_parallel(X)
    wT,time3 = transpose_parallel(w)
    dw,time4 = mul.matrix_mul_tile(XT, dout,tile_x,tile_y)
    dx,time5 = mul.matrix_mul_tile(dout, wT,tile_x,tile_y)
    start = time.time()
    dx = dx.reshape(x.shape)
    time6 = time.time()-start
    
    return dx, dw, db, time1+time2+time3+time4+time5+time6


def relu_forward(x):
    
    out = np.copy(x)
    out[np.where(out<0)] = 0
    
    return out

def relu_backward(dout, x):
    
    dx = np.copy(dout)
    dx[np.where(x<0)] = 0
    
    return dx

def softmax_loss(x, y):
    
    # Initialize the loss.
    loss = 0.0
    dx = np.zeros_like(x)
    
    num_train = x.shape[0]
    max_x = np.max(x,axis = 1).reshape(num_train,1)
    x_stability = x - max_x
    temp1 = np.exp(x_stability)/np.sum(np.exp(x_stability),axis=1).reshape(num_train,1)
    temp2 = np.zeros(temp1.shape).astype('float')
    temp2[np.arange(num_train).reshape(num_train,1),y.reshape(num_train,1)] = 1
    loss = -np.sum(temp2*np.log(temp1))
    dx = temp1 - temp2
    loss /= num_train
    dx /= num_train
    
    return loss, dx

class FullyConnectedLayer(object):
    def reset_layer(self, weight_scale=1e-2):
        """
        Reset weight and bias.
        
        Inputs:
        - weight_scale: (float) define the scale of weights
        """
        input_dim = self.input_dim
        hidden_dim = self.output_dim
        
        W = np.random.rand(input_dim, output_dim)
        b = np.zeros(output_dim)
        self.params = [W, b]
    
    def update_layer(self, params):
        """
        Update weight and bias
        """
        self.params = params

    
class DenseLayer(FullyConnectedLayer):
    """
    A dense hidden layer performs an affine transform followed by ReLU.
    Here we use ReLU as default activation function.
    """
    def __init__(self, input_dim, output_dim=100, weight_scale=1e-2):
        """
        Initialize weight W with random value and 
        bias b with zero.
        
        Inputs:
        - input_dim: (int) the number of input neurons, 
                     like D or D1xD2x...xDn.
        - output_dim: (int) the number of hidden neurons 
                      in this layer
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X = None
        self.A = None
        
        W = weight_scale*np.random.rand(input_dim, output_dim)
        b = np.zeros(output_dim)
        self.params = [W, b]
    
    def feedforward_serial(self, X):
        
        W, b = self.params
        self.X = X
        self.A = affine_forward_serial(X,W,b)
        out = relu_forward(self.A)
        
        return out
    
    def feedforward_parallel_tile(self, X,tile_x,tile_y):
        
        start = time.time()
        W, b = self.params
        self.X = X
        time1 = time.time()-start
        self.A,time2 = affine_forward_parallel_tile(X,W,b,tile_x,tile_y)
        start = time.time()
        out = relu_forward(self.A)
        time3 = time.time()-start
        
        return out, time1+time2+time3

    
    def backward_serial(self, dout):
        
        W, b = self.params
        X = self.X # cache input data
        A = self.A # cache intermediate affine result
        
        dX = np.zeros_like(X)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        dA = relu_backward(dout, A)
        dX, dW, db = affine_backward_serial(dA, X, W, b)
        
        self.gradients = [dW, db]
        
        return dX
    
    def backward_parallel_tile(self, dout,tile_x,tile_y):
        
        start = time.time()
        
        W, b = self.params
        X = self.X # cache input data
        A = self.A # cache intermediate affine result
        
        dX = np.zeros_like(X)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        dA = relu_backward(dout, A)
        
        time1 = time.time()-start
        
        dX, dW, db, time2 = affine_backward_parallel_tile(dA, X, W, b,tile_x,tile_y)
        
        start = time.time()
        self.gradients = [dW, db]
        time3 = time.time()-start
        
        return dX, time1+time2+time3
    
    
class AffineLayer(FullyConnectedLayer):
    
    def __init__(self, input_dim, output_dim=100, weight_scale=1e-2):
        """
        Initialize weight W with random value and 
        bias b with zero.
        
        Inputs:
        - input_dim: (int) the number of input neurons, 
                     like D or D1xD2x...xDn.
        - output_dim: (int) the number of hidden neurons in this layer
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        W = weight_scale*np.random.rand(input_dim, output_dim)
        b = np.zeros(output_dim)
        self.params = [W, b]
        self.X = None
    
    def feedforward_serial(self, X):
        
        W, b = self.params
        self.X = X
        
        out = affine_forward_serial(X, W, b)
        
        return out
    
    def feedforward_parallel_tile(self, X,tile_x,tile_y):
        
        start = time.time()
        W, b = self.params
        self.X = X
        time1 = time.time()-start
        out,time2 = affine_forward_parallel_tile(X, W, b,tile_x,tile_y)
        
        return out,time1+time2
   
    def backward_serial(self, dout):
        
        W, b = self.params
        X = self.X
        
        dX = np.zeros_like(X)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        
        dX, dW, db = affine_backward_serial(dout, X, W, b)
        
        self.gradients = [dW, db]
        
        return dX
    
    def backward_parallel_tile(self, dout,tile_x,tile_y):
        
        start = time.time()
        W, b = self.params
        X = self.X
        
        dX = np.zeros_like(X)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        
        time1 = time.time()-start
        
        dX, dW, db,time2 = affine_backward_parallel_tile(dout, X, W, b,tile_x,tile_y)
        
        start = time.time()
        self.gradients = [dW, db]
        time3 = time.time()-start
        
        return dX, time1+time2+time3
   


class TwoLayerNet(object):
    
    def __init__(self, input_dim=3072, hidden_dim=200, num_classes=10, reg=0.0, weight_scale=1e-2):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.layer1 = DenseLayer(input_dim, hidden_dim, weight_scale=weight_scale)
        self.layer2 = AffineLayer(hidden_dim, num_classes, weight_scale=weight_scale)
        self.reg = reg
        self.v = [0 ,0 ,0 ,0]

  
        
    def loss_serial(self, X, y):
        
        loss = 0.0
        reg = self.reg
        
        A = self.layer1.feedforward_serial(X)
        out = self.layer2.feedforward_serial(A)
        
        loss, dout = softmax_loss(out, y)
        dA = self.layer2.backward_serial(dout)
        dX = self.layer1.backward_serial(dA)
        
        # Add L2 regularization
        square_weights = np.sum(self.layer1.params[0]**2) + np.sum(self.layer2.params[0]**2)
        loss += 0.5*self.reg*square_weights
        return loss

    def loss_parallel_tile(self, X, y,tile_x,tile_y):
        
        loss = 0.0
        reg = self.reg
        
        A,time1 = self.layer1.feedforward_parallel_tile(X,tile_x,tile_y)
        out,time2 = self.layer2.feedforward_parallel_tile(A,tile_x,tile_y)
        
        start = time.time()
        loss, dout = softmax_loss(out, y)
        time3 = time.time()-start
        
        dA,time4 = self.layer2.backward_parallel_tile(dout,tile_x,tile_y)
        dX,time5 = self.layer1.backward_parallel_tile(dA,tile_x,tile_y)
        
        # Add L2 regularization
        start = time.time()
        square_weights = np.sum(self.layer1.params[0]**2) + np.sum(self.layer2.params[0]**2)
        loss += 0.5*self.reg*square_weights
        time6 = time.time()-start
        
        return loss, time1+time2+time3+time4+time5+time6
    

    

INPUT_DIM = 800
NUM_CLASSES = 10
REG = 1e-4
WEIGHT_SCALE = 1e-3
BATCH_SIZE = 32

serial_time = []
parallel_time_tile1 = []
parallel_time_tile2 = []
parallel_time_tile3= []
y_time=[]
flag = 0


HIDDEN_DIM = 200

length=32
x_label = np.array(range(1, length+1))
for i in range(length):
    tile_x=np.int((i+1))
    tile_y=np.int((i+1))
    test_data = np.random.rand(BATCH_SIZE, INPUT_DIM)
    test_label = np.random.randint(low=0, high=NUM_CLASSES - 1, size=BATCH_SIZE)

    model = TwoLayerNet(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, REG, WEIGHT_SCALE)

    loss_parallel_tile1, ptime_tile1 = model.loss_parallel_tile(test_data, test_label,tile_x,tile_y)
    
    parallel_time_tile1.append(ptime_tile1)

plt.figure()  # draw results
plt.title('tile size effects on running time')
plt.plot(x_label, parallel_time_tile1)
plt.xlabel('tile size')
plt.ylabel('run time/s')
plt.legend(loc='upper left')
plt.savefig('loop_tile.png')
   
