import numpy as np
import learner as ln
from learner.utils import mse, grad
import h5py
import torch
import matplotlib.pyplot as plt

# 准备PINN的数据集。
class NSData(ln.Data):
    def __init__(self, train_num):
        super(NSData, self).__init__()
        self.train_num = train_num
        self.__init_data()

    def generate(self, num):
        f = h5py.File('./data_16.h5', 'r')
        #n_frame = len(f.keys())
        tlist = ["012.6", "012.9", "013.2", "013.5"]
        n_frame = len(tlist)

        XYT = np.zeros((0,3))
        UX = np.zeros((0,1))
        UY = np.zeros((0,1))
        P = np.zeros((0,1))

        #for key in f.keys():
        for key in tlist:
            time = float(key)
            XY = f[key][:,0:2]
            ux = f[key][:,2][:,None]
            uy = f[key][:,3][:,None]
            p  = f[key][:,4][:,None]
            XYT = np.concatenate((XYT, np.concatenate((XY, np.full((XY.shape[0],1), time)), axis=1)), axis=0)
            UX  = np.concatenate((UX, ux), axis=0)
            UY  = np.concatenate((UY, uy), axis=0)
            P   = np.concatenate((P, p), axis=0)

        X, y = {}, {}
        randXY = np.random.rand(num['pde']*n_frame, 2)*2*np.pi-np.pi
        randT = np.random.rand(num['pde']*n_frame, 1) * (float(tlist[-1]) - float(tlist[0])) + float(tlist[0])

        X['pde'] = np.concatenate((randXY, randT), axis=1)
        X['data'] = XYT
        np.savetxt("aaa.data", X['pde'])
        plt.scatter(X['data'][:,0], X['data'][:,1])
        plt.show()
        plt.scatter(X['data'][:,1], X['data'][:,2])
        plt.show()
        y['data'] = np.concatenate((UX, UY, P), axis=1)
        return X, y

    def __init_data(self):
        self.X_train, self.y_train = self.generate(self.train_num)
        self.X_test, self.y_test = self.X_train, self.y_train

# 定义PINN中用到的方程，此处即为NS方程。
class PINN(ln.nn.Algorithm):
    def __init__(self, net, lam=1):
        super(PINN, self).__init__()
        self.net = net
        self.lam = lam

    def criterion(self, X, y):
        z = X['pde'].requires_grad_(True)
        u = self.net(z)
        vx, vy = u[:,0], u[:,1]
        vx_g = grad(u[:,0].reshape(-1,1), z)
        vy_g = grad(u[:,1].reshape(-1,1), z)
        p_g = grad(u[:,2].reshape(-1,1), z)

        vx_x, vx_y, vx_t = vx_g[:, 0], vx_g[:, 1], vx_g[:, 2]
        vy_x, vy_y, vy_t = vy_g[:, 0], vy_g[:, 1], vy_g[:, 2]

        vx_xx, vx_yy = grad(vx_x.reshape(-1,1), z)[:, 0], grad(vx_y.reshape(-1,1), z)[:, 1]
        vy_xx, vy_yy = grad(vy_x.reshape(-1,1), z)[:, 0], grad(vy_y.reshape(-1,1), z)[:, 1]
        p_x, p_y = p_g[:, 0], p_g[:, 1]

        fx = 0.025 * torch.sin(2*X['pde'][:,0]) * torch.cos(2*X['pde'][:,1])
        fy = -0.025 * torch.cos(2*X['pde'][:,0]) * torch.sin(2*X['pde'][:,1])
        # e_pde。由三部分组成，分别是三个控制方程的残差。
        MSEr1 = mse(vx_t + vx*vx_x + vy*vx_y, -p_x + 4.66e-4*(vx_xx + vx_yy) + fx)
        MSEr2 = mse(vy_t + vx*vy_x + vy*vy_y, -p_y + 4.66e-4*(vy_xx + vy_yy) + fy)
        MSEr3 = mse(vx_x, -vy_y)
        # e_data。
        MSEd = mse(self.net(X['data']), y['data'])
        # e_PINN = e_pde + lam*e_data.
        return MSEr1 + MSEr2 + MSEr3 + self.lam * MSEd

    def predict(self, x, returnnp=False):
        return self.net.predict(x, returnnp)

# 推理全场高精度512X512的解。
def predict(net):
    #t = np.linspace(0,30,101)
    t = ["012.6", "012.9", "013.2", "013.5"]
    nn = 512 * 512
    d = 2*np.pi/512
    x = np.linspace(-np.pi, np.pi-d, 512)
    y = np.linspace(-np.pi, np.pi-d, 512)
    coordinates = [[xi, yi] for xi in x for yi in y]
    frame_data = np.zeros((nn,5))
    writeH5File = h5py.File('Prediction.h5', 'w')
    for time in t:
        for i in range(0, nn):
            X = coordinates[i][0]
            Y = coordinates[i][1]
            XYT = np.array([X, Y, float(time)])
            predict = net.predict(XYT, returnnp=True)
            frame_data[i,:] = np.concatenate((X.reshape(1,-1), Y.reshape(1,-1), predict[None,:]), axis=1)

        fig, ax = plt.subplots()
        scatter = ax.scatter(np.array(coordinates)[:,0], np.array(coordinates)[:,1], c=np.sqrt(frame_data[:,2]**2+frame_data[:,3]**2), s=1, vmin=0, vmax=0.4)
        ax.set_aspect('equal')
        cbar = plt.colorbar(scatter)
        ax.set_title("t={}".format(time))
        fig.savefig("fig.{}.png".format(time), dpi=300)
        fig.show()

        writeH5File.create_dataset('{:0>5.1f}'.format(float(time)), data=frame_data)
    writeH5File.close()


def main():
    device = 'gpu' # 'cpu' or 'gpu'

    # 全场布置的P_pde的点的个数。默认位置随机分布。
    train_num = {'pde': 3000}

    # PINN的网络结构，默认采用FNN。输入为3维:[x,y,t]，输出为3维:[Ux,Uy,P].
    size = [3, 100, 100, 100, 100, 3]
    activation = 'sigmoid'
    lr = 0.0005
    iterations = 20000
    print_every = 10
    batch_size = None

    data = NSData(train_num)
    fnn = ln.nn.FNN(size, activation)
    net = PINN(fnn)
    args = {
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device,
    }

    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()

    predict(ln.Brain.Best_model())


if __name__ == '__main__':
    main()
