import pdb, sys, traceback, os
import numpy as np
import pickle as pkl
import scipy as sp
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import hickle as hkl
from scipy.io import loadmat

def generate_random_face(save_name, gender):
	if save_name[-3:]!='.fg':
		save_name += '.fg'
	os.system('fg3 controls '+save_name+' random all '+gender)

def facegen_construct(fg_file, save_name):
	os.system('fg3 construct /home/bill/Dropbox/Research/Libraries/FaceGen/SDK/data/fg3/sdk/csamDefault33/HeadHires '+fg_file+' '+save_name)

def facegen_render(xml_file):
	os.system('fg3 render '+xml_file)

def get_age_from_fg(fg_file):
	tmp_file_name = '/home/bill/Projects/Predictive_Networks/misc/ageshape.txt'
	os.system('fg3 controls '+fg_file+' age shape > '+tmp_file_name)
	f = open(tmp_file_name, 'r')
	lines = f.read().splitlines()
	f.close()
	return float(lines[1][11:16])

def get_gender_from_fg(fg_file):
	tmp_file_name = '/home/bill/Projects/Predictive_Networks/misc/gendershape.txt'
	os.system('fg3 controls '+fg_file+' gender shape > '+tmp_file_name)
	f = open(tmp_file_name, 'r')
	lines = f.read().splitlines()
	f.close()
	gender = float(lines[1][lines[1].find('(')+1:lines[1].find('(')+5])
	return gender

def get_basis_from_fg(fg_file):
	tmp_file_name = '/home/bill/Projects/Predictive_Networks/misc/pcabasis.txt'
	os.system('fg3 coord '+fg_file+' > '+tmp_file_name)
	f = open(tmp_file_name, 'r')
	lines = f.read().splitlines()
	f.close()
	vals = np.zeros(len(lines)-3)
	for idx,i in enumerate(range(2,len(lines)-1)):
		vals[idx] = float(lines[i][lines[i].find(':')+2:lines[i].find(':')+8])
	return vals


def create_clipset():

	clipset_num = 17
	P = {}
	P['n_train'] = 4000
	P['n_val'] = 200
	P['n_test'] = 200
	P['n_clips'] = P['n_train']+P['n_val']+P['n_test']
	P['n_frames'] = 6
	P['pan_initial_angle_range'] = (-np.pi/2, np.pi/2)
	P['pan_angular_speed_range'] = 0 #(-np.pi/6, np.pi/6)
	#P['tilt_initial_angle_range'] = (-np.pi/4, np.pi/4)
	#P['tilt_angular_speed_range'] = (-np.pi/8, np.pi/8)
	#P['roll_initial_angle_range'] = (-np.pi/4, np.pi/4)
	#P['roll_angular_speed_range'] = (-np.pi/8, np.pi/8)
	P['clipset_num'] = clipset_num
	P['render_file'] = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/render.xml'


	base_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/'
	if not os.path.exists(base_dir):
		os.mkdir(base_dir)

	f = open(P['render_file'], 'r')
	lines = f.readlines()
	f.close()

	pan_initial_angles = np.zeros(P['n_clips'])
	pan_angular_speeds = np.zeros(P['n_clips'])

	for i in range(P['n_clips']):
		fg_file = base_dir+'fg_files/face_'+str(i)+'.fg'
		c_file = base_dir+'construct_files/face_'+str(i)
		im_file = base_dir+'images/face_'+str(i)
		if np.random.uniform()<0.5:
			gender='male'
		else:
			gender='female'
		print 'Clip '+str(i)+' '+gender
		generate_random_face(fg_file, gender)
		facegen_construct(fg_file, c_file)
		pan_theta0 = np.random.uniform(P['pan_initial_angle_range'][0], P['pan_initial_angle_range'][1])
		if isinstance(P['pan_angular_speed_range'], float):
			pan_alpha = P['pan_angular_speed_range']
		else:
			pan_alpha = np.random.uniform(P['pan_angular_speed_range'][0], P['pan_angular_speed_range'][1])
		#tilt_theta0 = np.random.uniform(P['tilt_initial_angle_range'][0], P['tilt_initial_angle_range'][1])
		#tilt_alpha = np.random.uniform(P['tilt_angular_speed_range'][0], P['tilt_angular_speed_range'][1])
		#roll_theta0 = np.random.uniform(P['roll_initial_angle_range'][0], P['roll_initial_angle_range'][1])
		#roll_alpha = np.random.uniform(P['roll_angular_speed_range'][0], P['roll_angular_speed_range'][1])

		lines[8] = '\t\t\t<triFilename>'+c_file+'.tri</triFilename>\n'
		lines[9] = '\t\t\t<imgFilename>'+c_file+'.bmp</imgFilename>\n'

		pan_initial_angles[i] = pan_theta0
		pan_angular_speeds[i] = pan_alpha


		for j in range(P['n_frames']):
			#lines[26] = '\t\t<rollRadians>'+str(roll_theta0+j*roll_alpha)+'</rollRadians>\n'
			#lines[27] = '\t\t<tiltRadians>'+str(tilt_theta0+j*tilt_alpha)+'</tiltRadians>\n'
			lines[28] = '\t\t<panRadians>'+str(pan_theta0+j*pan_alpha)+'</panRadians>\n'
			lines[82] = '\t<outputFile>'+im_file+'_frame_'+str(j)+'.png</outputFile>\n'
			f = open(base_dir+'tmp_render.xml', 'w')
			f.writelines(lines)
			f.close()
			facegen_render(base_dir+'tmp_render')

	f = open(base_dir+'params.pkl','w')
	pkl.dump({'P': P, 'pan_initial_angles': pan_initial_angles, 'pan_angular_speeds': pan_angular_speeds}, f)
	f.close()



def create_decoding_clipset():

	clipset_num = 14
	is_static = True

	P = {}
	P['n_faces'] = 500
	P['is_static'] = is_static
	if is_static:
		P['n_angles'] = 12
		P['pan_angles'] = np.linspace(-np.pi/2, np.pi/2, P['n_angles'])
		P['n_frames'] = 1
	else:
		P['n_speeds'] = 10
		P['pan_angular_speeds'] = np.linspace(-np.pi/6, np.pi/6, P['n_speeds'])
		P['n_frames'] = 6
		P['pan_initial_angle_range'] = (-np.pi/2, np.pi/2)
	P['clipset_num'] = clipset_num
	P['render_file'] = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/render.xml'

	if is_static:
		ind_var = 'n_angles'
	else:
		ind_var = 'n_speeds'

	face_labels = np.zeros(P['n_faces']*P[ind_var], int)
	pan_angular_speeds = np.zeros(P['n_faces']*P[ind_var])
	pan_initial_angles = np.zeros(P['n_faces']*P[ind_var])
	if is_static:
		pan_angle_labels = np.zeros(P['n_faces']*P[ind_var])
	else:
		pan_speed_labels = np.zeros(P['n_faces']*P[ind_var])

	base_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/'
	if not os.path.exists(base_dir):
		os.mkdir(base_dir)

	f = open(P['render_file'], 'r')
	lines = f.readlines()
	f.close()

	count = 0
	for i in range(P['n_faces']):
		fg_file = base_dir+'fg_files/face_'+str(i)+'.fg'
		c_file = base_dir+'construct_files/face_'+str(i)
		if np.random.uniform()<0.5:
			gender='male'
		else:
			gender='female'
		print 'Clip '+str(i)+' '+gender
		generate_random_face(fg_file, gender)
		facegen_construct(fg_file, c_file)

		lines[8] = '\t\t\t<triFilename>'+c_file+'.tri</triFilename>\n'
		lines[9] = '\t\t\t<imgFilename>'+c_file+'.bmp</imgFilename>\n'

		for j in range(P[ind_var]):
			if is_static:
				pan_theta0 = P['pan_angles'][j]
				pan_alpha = 0.0
			else:
				pan_theta0 = np.random.uniform(P['pan_initial_angle_range'][0], P['pan_initial_angle_range'][1])
				pan_alpha = P['pan_angular_speeds'][j]

			pan_initial_angles[count] = pan_theta0
			pan_angular_speeds[count] = pan_alpha
			face_labels[count] = i
			if is_static:
				pan_angle_labels[count] = j
			else:
				pan_speed_labels[count] = j
			count += 1

			im_file = base_dir+'images/face_'+str(i)+'_rotation_'+str(j)

			for k in range(P['n_frames']):
				lines[28] = '\t\t<panRadians>'+str(pan_theta0+k*pan_alpha)+'</panRadians>\n'
				lines[82] = '\t<outputFile>'+im_file+'_frame_'+str(k)+'.png</outputFile>\n'
				f = open(base_dir+'tmp_render.xml', 'w')
				f.writelines(lines)
				f.close()
				facegen_render(base_dir+'tmp_render')

	if is_static:
		out_dict = {'P': P, 'pan_initial_angles': pan_initial_angles, 'pan_angular_speeds': pan_angular_speeds, 'face_labels': face_labels, 'pan_angle_labels': pan_angle_labels}
	else:
		out_dict = {'P': P, 'pan_initial_angles': pan_initial_angles, 'pan_angular_speeds': pan_angular_speeds, 'face_labels': face_labels, 'pan_speed_labels': pan_speed_labels}

	f = open(base_dir+'params.pkl','w')
	pkl.dump(out_dict, f)
	f.close()


def create_numpy_clipset():

	clipset_num = 5
	f = open('/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/params.pkl', 'r')
	d = pkl.load(f)
	P = d['P']
	f.close()

	for t in ['train', 'val', 'test']:

		clips = np.zeros((P['n_'+t], P['n_frames'], 1, 150, 150)).astype(np.float32)

		if t=='train':
			offset = 0
		elif t=='val':
			offset = P['n_train']
		else:
			offset = P['n_train']+P['n_val']

		for i in range(P['n_'+t]):
			for j in range(P['n_frames']):
				im_file = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/images_processed/face_'+str(i+offset)+'_frame_'+str(j)+'.png'
				im = imread(im_file).astype(np.float32)/65535
				clips[i,j,0] = im

		f = open('/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/clips'+t+'.hkl', 'w')
		hkl.dump(clips, f)
		f.close()

def create_numpy_clipset_decoding():

	clipset_num = 14
	f = open('/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/params.pkl', 'r')
	d = pkl.load(f)
	P = d['P']
	f.close()

	if 'is_static' in P:
		is_static = P['is_static']
	elif P['n_frames']==1:
		is_static = True
	else:
		is_static = False

	if is_static:
		n_frames = 6
		clips = np.zeros((P['n_faces']*P['n_angles'], n_frames, 1, 150, 150)).astype(np.float32)

		count = -1
		for i in range(P['n_faces']):
			for k in range(P['n_angles']):
				count += 1
				im_file = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/images_processed/face_'+str(i)+'_rotation_'+str(k)+'_frame_'+str(0)+'.png'
				im = imread(im_file).astype(np.float32)/65535
				clips[count,:,0] = im
	else:
		clips = np.zeros((P['n_faces']*P['n_speeds'], P['n_frames'], 1, 150, 150)).astype(np.float32)

		count = -1
		for i in range(P['n_faces']):
			for k in range(P['n_speeds']):
				count += 1
				for j in range(P['n_frames']):
					im_file = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/images_processed/face_'+str(i)+'_rotation_'+str(k)+'_frame_'+str(j)+'.png'
					im = imread(im_file).astype(np.float32)/65535
					clips[count,j,0] = im

	f = open('/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/clipsall.hkl', 'w')
	hkl.dump(clips, f)
	f.close()


def create_clipset_weights():

	clipset_num = 3
	n_size = 3
	compression_factor = 3
	min_weight = 0.1

	f = open('/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/params.pkl', 'r')
	d = pkl.load(f)
	P = d['P']
	f.close()

	weights = np.zeros( (P['n_clips'], P['n_frames'], 150, 150) ).astype(np.float32)

	for i in range(P['n_clips']):
		for j in range(P['n_frames']):
			im_file = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/filtered_images_stdfilt_n'+str(n_size)+'_p'+str(compression_factor)+'/face_'+str(i)+'_frame_'+str(j)+'.png'
			im = imread(im_file).astype(np.float32)
			weights[i,j] = im

	# plt.hist(weights.reshape(np.prod(weights.shape)))
	# plt.show(block=False)
	# pdb.set_trace()

	weights = weights/np.mean(weights)
	weights[weights<min_weight] = min_weight
	print 'Mean weight: '+str(np.mean(weights))
	weights = weights.reshape( (P['n_clips'], P['n_frames'], 150*150) )

	for t in ['train','val','test']:
		if t=='train':
			w_start = 0
			w_end = P['n_train']
		elif t=='val':
			w_start = P['n_train']
			w_end = P['n_train']+P['n_val']
		else:
			w_start = P['n_train']+P['n_val']
			w_end = P['n_clips']

		f = open('/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/weights'+t+'_n'+str(n_size)+'_p'+str(compression_factor)+'_minw'+str(min_weight)+'.hkl', 'w')
		hkl.dump(weights[w_start:w_end], f)
		f.close()


def create_features(clipset_num):

	base_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/'
	f = open(base_dir+'params.pkl','r')
	d = pkl.load(f)
	P = d['P']
	f.close()

	for t in ['train', 'val', 'test']:
		print 'Creating features: '+t

		if t=='train':
			offset = 0
		elif t=='val':
			offset = P['n_train']
		else:
			offset = P['n_train']+P['n_val']
		n_clips = d['P']['n_'+t]

		ages = np.zeros(n_clips)
		genders = np.zeros(n_clips)
		pca_basis = np.zeros((n_clips,130))

		for i in range(n_clips):
			f_name = base_dir+'fg_files/face_'+str(i+offset)+'.fg'
			genders[i] = get_gender_from_fg(f_name)
			ages[i] = get_age_from_fg(f_name)
			pca_basis[i] = get_basis_from_fg(f_name)

		f = open(base_dir+'face_params_'+t+'.pkl','w')
		pkl.dump({'genders': genders, 'ages': ages, 'pca_basis': pca_basis}, f)
		f.close()

def create_features_decoding(clipset_num):

	base_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/'
	f = open(base_dir+'params.pkl','r')
	d = pkl.load(f)
	P = d['P']
	f.close()


	ages = np.zeros(d['P']['n_faces'])
	genders = np.zeros(d['P']['n_faces'])
	pca_basis = np.zeros((d['P']['n_faces'],130))

	for i in range(d['P']['n_faces']):
		f_name = base_dir+'fg_files/face_'+str(i)+'.fg'
		genders[i] = get_gender_from_fg(f_name)
		ages[i] = get_age_from_fg(f_name)
		pca_basis[i] = get_basis_from_fg(f_name)

	f = open(base_dir+'face_params.pkl','w')
	pkl.dump({'genders': genders, 'ages': ages, 'pca_basis': pca_basis}, f)
	f.close()




def create_combined_features(clipset_num):

	base_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/'
	f = open(base_dir+'params.pkl','r')
	params_dict = pkl.load(f)
	P = params_dict['P']
	f.close()

	for t in ['train', 'val', 'test']:

		if t=='train':
			offset = 0
		elif t=='val':
			offset = P['n_train']
		else:
			offset = P['n_train']+P['n_val']
		n_clips = params_dict['P']['n_'+t]


		f = open(base_dir+'face_params_'+t+'.pkl','r')
		face_params = pkl.load(f)
		f.close()

		out_dict = {}
		for key in face_params:
			out_dict[key] = face_params[key]

		for key in params_dict.keys():
			if key!='P':
				out_dict[key] = params_dict[key][offset:offset+n_clips]

		if os.path.exists(base_dir+'postprocess_params.mat'):
			pp_dict = loadmat(base_dir+'postprocess_params.mat')

			for key in pp_dict.keys():
				if key!='P':
					out_dict[key] = pp_dict[key][offset:offset+n_clips]

		f = open(base_dir+'all_params_'+t+'.pkl','w')
		pkl.dump(out_dict, f)
		f.close()

	for t in ['train', 'val', 'test']:
		f = open(base_dir+'all_params_'+t+'.pkl','r')
		this_dict = pkl.load(f)
		f.close()
		if t=='train':
			out_dict = this_dict
		else:
			for key in out_dict:
				out_dict[key] = np.concatenate( (out_dict[key], this_dict[key]), axis=0 )
	f = open(base_dir+'all_params_all.pkl','w')
	pkl.dump(out_dict, f)
	f.close()


def concatenate_clips(clipset_num):

	base_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/'
	for i,t in enumerate(['train', 'val', 'test']):
		if i==0:
			clips = hkl.load( open(base_dir+'clips'+t+'.hkl', 'r'))
		else:
			these_clips = hkl.load( open(base_dir+'clips'+t+'.hkl', 'r'))
			clips = np.concatenate( (clips, these_clips), axis=0)
	pdb.set_trace()
	hkl.dump(clips, open(base_dir+'clipsall.hkl','w'))


def make_static_dataset():

	orig_clipset = 5
	new_clipset = 18
	rep_every_frame = False

	orig_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(orig_clipset)+'/'
	out_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(new_clipset)+'/'
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	d = pkl.load(open(orig_dir+'params.pkl', 'r'))
	P = d['P']
	if rep_every_frame:
		angles = np.zeros( (d['pan_angular_speeds'].shape[0],P['n_frames']) )
		for t in range(P['n_frames']):
			a = d['pan_initial_angles']+t*d['pan_angular_speeds']
			for i in range(len(a)):
				while a[i]>2*np.pi:
					a[i] -= 2*np.pi
				while a[i]<0:
					a[i] += 2*np.pi
			angles[:,t] = a
		d['pan_initial_angles'] = angles.flatten()
		d['pan_angular_speeds'] = np.zeros(d['pan_angular_speeds'].shape[0]*P['n_frames'])
	else:
		d['pan_angular_speeds'] = 0
	pkl.dump(d, open(out_dir+'params.pkl','w'))

	for t in ['train', 'val', 'test']:

		if rep_every_frame:
			clips = np.zeros((P['n_'+t]*P['n_frames'], P['n_frames'], 1, 150, 150)).astype(np.float32)
		else:
			clips = np.zeros((P['n_'+t], P['n_frames'], 1, 150, 150)).astype(np.float32)

		old_clips = hkl.load(open(orig_dir+'clips'+t+'.hkl','r'))
		if rep_every_frame:
			for j in range(P['n_frames']):
				for k in range(P['n_frames']):
					clips[j*P['n_'+t]:(j+1)*P['n_'+t],k,0] = old_clips[:,j,0]
		else:
			for j in range(P['n_frames']):
				clips[:,j,0] = old_clips[:,0,0]

		f = open(out_dir+'clips'+t+'.hkl', 'w')
		hkl.dump(clips, f)
		f.close()



# pc num is zero indexed
def morph_face(orig_fg_file, out_fg_file, pc_num, pc_val):

	os.system('cp '+orig_fg_file+' '+out_fg_file)
	os.system('fg3 coord '+out_fg_file+' '+str(pc_num)+' '+str(pc_val))


def morph_test():
	clipset_num=10
	pcs = [0,1,2]
	base_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset_num)+'/'
	out_dir = base_dir +'morph_test/'

	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	im_dir = out_dir + 'ims/'
	if not os.path.exists(im_dir):
		os.mkdir(im_dir)

	f = open(base_dir+'render.xml', 'r')
	lines = f.readlines()
	f.close()


	for pc_num in pcs:
		for face in range(10):
			orig_fg_file = base_dir+'fg_files/face_'+str(face)+'.fg'
			for ki,k in enumerate(np.linspace(-3, 3,12)):
				out_file = out_dir+'face_'+str(face)+'_'+str(ki)+'.fg'
				morph_face(orig_fg_file, out_file, pc_num, k)
				out_construct = out_dir+'face_'+str(face)+'_'+str(ki)
				facegen_construct(out_file, out_construct)
				lines[8] = '\t\t\t<triFilename>'+out_construct+'.tri</triFilename>\n'
				lines[9] = '\t\t\t<imgFilename>'+out_construct+'.bmp</imgFilename>\n'
				lines[28] = '\t\t<panRadians>'+str(np.pi/4)+'</panRadians>\n'
				lines[82] = '\t<outputFile>'+im_dir+'pc'+str(pc_num+1)+'_face_'+str(face)+'_'+str(ki)+'.png</outputFile>\n'
				f = open(base_dir+'tmp_render.xml', 'w')
				f.writelines(lines)
				f.close()
				facegen_render(base_dir+'tmp_render')


if __name__=='__main__':
	try:
		#create_clipset()
		#create_numpy_clipset()
		#create_features(5)
		#create_combined_features(5)
		#create_clipset_weights()
		concatenate_clips(18)
		#create_decoding_clipset()
		#create_numpy_clipset_decoding()
		#make_static_dataset()
		#morph_test()
		#create_features_decoding(10)

	except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
