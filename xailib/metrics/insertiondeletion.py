from xailib.models.bbox import AbstractBBox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import torch
import matplotlib.animation as animation
from matplotlib import rc
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**32

class ImageInsDel():
    def __init__(self, predict, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.

        Args:
            predict (func): function that takes in input a numpy array and return the prediction.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.predict = predict
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def __call__(self, img, size, explanation, rgb=True, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.

        Args:
            img (np.ndarray): normalized image tensor.
            size (int): size of the image ex:224
            explanation (np.ndarray): saliency map.
            rgb (bool): if the image is rgb or grayscale
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        if rgb:
            CH = 3
        else: 
            CH = 1
        HW = size * size # image area
        pred = torch.tensor(self.predict(img))
        top, c = torch.max(pred, 1)
        c = c[0]
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion metric'
            xlabel = 'Pixels deleted'
            start = torch.tensor(img).clone()
            finish = self.substrate_fn(torch.tensor(img))
        elif self.mode == 'ins':
            title = 'Insertion metric'
            xlabel = 'Pixels inserted'
            start = self.substrate_fn(torch.tensor(img))
            finish = torch.tensor(img).clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)

        if verbose:
            fig, ax = plt.subplots(1,2,figsize=(10,5),facecolor='white')
            
            #ax[0].axis('off')
            ax[1].set_title(title)
            ax[1].set_xlabel(xlabel)
            ax[1].set_xlim(-0.1, 1.1)
            ax[1].set_ylim(0, 1.05)
            line = ax[1].plot([],[],'-o')[0]
            title = ax[0].set_title('')
            probs = ax[0].set_xlabel('')
            title.set_animated(True)
            probs.set_animated(True)
            plt.close()

            def init():
                line.set_data([], [])
                title.set_text('')
                probs.set_text('')
                return line, title, probs

            def animate(i):
                pred = torch.tensor(self.predict(start.numpy()))
                pr, cl = torch.topk(pred, 2)
                probs.set_text(f'class {cl[0][0]} prob {float(pr[0][0]):.3f} \n class {cl[0][1]} prob {float(pr[0][1]):.3f}')
                scores[i] = pred[0, c]
                title.set_text('{} {:.1f}%, P={:.4f}'.format(xlabel, 100 * i / n_steps, scores[i]))
                line.set_data(np.arange(i+1) / n_steps, scores[:i+1])
                image = start[0,:].detach().cpu().numpy()
                if rgb:
                    ax[0].imshow(image.transpose(1,2,0))
                else:
                    ax[0].imshow(image, cmap='gray')
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, CH, HW)[0, :, coords] = finish.cpu().numpy().reshape(1, CH, HW)[0, :, coords]
                return line, title, probs
            

            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=n_steps+1, interval=70, blit=True, repeat=False)

            rc('animation', html='jshtml')
            return [scores, anim]

        else:
            for i in range(n_steps+1):
                pred = torch.tensor(self.predict(start.numpy()))
                scores[i] = pred[0, c]
                if i < n_steps:
                    coords = salient_order[:, self.step * i : self.step * (i + 1)]
                    start.cpu().numpy().reshape(1, CH, HW)[0, :, coords] = finish.cpu().numpy().reshape(1, CH, HW)[0, :, coords]
            return scores