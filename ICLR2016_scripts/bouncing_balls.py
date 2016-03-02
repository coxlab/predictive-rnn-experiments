import numpy as np
import hickle as hkl
import pickle as pkl
import scipy.io as spio
import sys, traceback, pdb, os

class Ball:
    def __init__(self, radius, x0, v0):
        self.velocity = v0
        self.radius = radius
        self.position = x0
        self.mass = self.radius**2

    def update(self, balls):
        proposed_position = self.position + self.velocity




def create_clip_set():

    P = {}
    P['screen_size'] = 64
    P['n_clips'] = 500
    P['nt'] = 128
    P['max_v'] = 4
    P['min_v'] = 1
    P['min_r'] = 3
    P['max_r'] = 7
    P['is_square'] = True

    save_folder = '/home/bill/Data/Bouncing_Balls/clip_set0/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    clips = np.zeros((P['n_clips'], P['nt'], P['screen_size'], P['screen_size']))

    #clip_info = {'x0': np.zeros((P['n_clips'], 2)), 'v': np.zeros(P['n_clips']), 'theta0': np.zeros(P['n_clips']), 'v0': np.zeros((P['n_clips'], 2)), 'r': np.zeros((P['n_clips']))}
    clip_info = {'x0': np.zeros((P['n_clips'], 2)), 'v0': np.zeros((P['n_clips'], 2)), 'r': np.zeros((P['n_clips']))}

    for i in range(P['n_clips']):
        print 'clip '+str(i)
        x0 = np.random.randint(P['screen_size']/4, 3*P['screen_size']/4, size=2)
        v = np.random.uniform(P['min_v'], P['max_v'])
        theta = np.random.uniform(0, 2*np.pi)
        v0 = np.array([int(np.round(v*np.cos(theta))), int(np.round(v*np.sin(theta)))])
        # v0 = np.random.randint(P['min_v'], P['max_v']+1, 2)
        # if np.sum(v0)==0:
        #     if np.random.uniform()<0.5:
        #         idx= 0
        #     else:
        #         idx = 1
        #     v0[idx] = np.random.randint(1, P['max_v']+1)
        r = np.random.randint(P['min_r'], P['max_r']+1)
        #clips[i] = create_ball_bouncing(x0, v0, r, P['nt'], P['screen_size'])
        clips[i] = create_ball_bouncing(x0, v0, r, P['nt'], P['screen_size'], P['is_square'])
        clip_info['x0'][i] = x0
        #clip_info['v'][i] = v
        #clip_info['theta0'][i] = theta
        clip_info['v0'][i] = v0
        clip_info['r'][i] = r

    f_name = save_folder + 'clips.hkl'
    f = open(f_name, 'w')
    hkl.dump(clips, f)
    f.close()

    f_name = save_folder + 'clip_info.pkl'
    f = open(f_name, 'w')
    pkl.dump([P, clip_info], f)
    f.close()

    spio.savemat(save_folder+'clips.mat', {'clips': clips[:10]})
    spio.savemat(save_folder+'clip_info.mat', clip_info)
    #spio.savemat('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/clip_info.mat', clip_info)




def create_ball_bouncing(x0, v0, r, nt, screen_size, is_square):

    frames = np.zeros((nt, screen_size, screen_size)).astype(np.float32)

    frames[0] = fill_frame(x0, r, frames[0], is_square)

    x = x0
    v = v0

    for i in range(1, nt):
        x,v = update_state(x, v, r, screen_size)
        frames[i] = fill_frame(x, r, frames[i], is_square)

    return frames


def fill_frame(x, r, frame, is_square):

    if is_square:
        for i in range(x[0]-r, x[0]+r+1):
            for j in range(x[1]-r, x[1]+r+1):
                frame[i,j] = 1
    else:
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                d = np.sqrt( (x[0]-i)**2 + (x[1]-j)**2)
                if d<= r:
                    frame[i,j] = 1

    return frame


def update_state(x, v, r, screen_size):

    x += v

    #while outside box, reflect across edge and flip velocity
    while x[0]-r<0 or x[0]+r>(screen_size-1) or x[1]-r<0 or x[1]+r>(screen_size-1):
        if x[0]-r<0:
            x[0]= 2*r-x[0]
            v[0] = -1*v[0]
        if x[0]+r>=(screen_size-1):
            x[0]= (screen_size-1)-r-(x[0]+r-(screen_size-1))
            v[0] = -1*v[0]
        if x[1]-r<0:
            x[1]= 2*r-x[1]
            v[1] = -1*v[1]
        if x[1]+r>=(screen_size-1):
            x[1]= (screen_size-1)-r-(x[1]+r-(screen_size-1))
            v[1] = -1*v[1]

    return x, v



if __name__=='__main__':
	try:
		create_clip_set()

	except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
