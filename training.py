import mido
from tqdm import tqdm
import os
import numpy as np
dir_folder='maestro-v2.0.0'
folders=next(os.walk(dir_folder))[1]

mids=[]

def clear_msgs(midi):
	lefttime=0
	for i in range(len(midi.tracks[1])):
		if midi.tracks[1][i].type=='control_change':
			lefttime+=midi.tracks[1][i].time
		else:
			midi.tracks[1][i].time+=lefttime
			lefttime=0

	for i in range(len(midi.tracks[1]))[::-1]:
		if midi.tracks[1][i].type=='control_change':
			midi.tracks[1].pop(i)
	return midi


for folder in tqdm(folders):
	for name in os.listdir('/'.join((dir_folder,folder))):
		mids.append(clear_msgs(mido.MidiFile('/'.join((dir_folder,folder,name)))))


def msg2array(msg):
	a=np.zeros(128+2+2)
	a[msg.note+2]=1
	a[-2]=msg.velocity/127
	a[-1]=msg.time
	return a

			
padded_mids=[]

for mid in tqdm(mids):
	st=np.zeros(130+2)
	st[0]=1
	sa=[st]
	for msg in mid.tracks[1]:
		if msg.type in ['note_on']:
			sa.append(msg2array(msg))
	en=np.zeros(130+2)
	en[1]=1
	sa.append(en)
	padded_mids.append(sa)


from keras.models import Model
from keras import layers as kl
from keras import backend as K
from keras import losses as kls

##############################
#here you can limit the gpu memory usage however its not necessary for the code to work
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
K.set_session(sess)
##############################

def myloss(y_true,y_pred):
	notes_pred=y_pred[:,:,:128+2]
	notes_true=y_true[:,:,:128+2]

	vel_pred=y_pred[:,:,128+2]
	vel_true=y_true[:,:,128+2]

	time_pred=y_pred[:,:,-1]
	time_true=y_true[:,:,-1]


	loss_notes=kls.categorical_crossentropy(notes_true,notes_pred)*1
	loss_vel=K.abs(vel_true-vel_pred)*1
	loss_time=((time_true-time_pred)**2)*0.01
	return loss_notes+loss_vel+loss_time

li=kl.Input(shape=(None,128+2+1+1))
l=li
for i in range(3):
	l=kl.CuDNNGRU(512,return_sequences=True)(l)
l=kl.Dense(512,activation='elu')(l)
ln=kl.Dense(128+2,activation='softmax')(l)
lv=kl.Dense(1,activation='sigmoid')(l)
lt=kl.Dense(1,activation='relu',use_bias=False)(l)
l=kl.Concatenate()([ln,lv,lt])
model=Model(li,l)
model.compile("adamax",myloss)

def generator(batch_size,seq_length):
	batch=[]
	while True:
		mid_index=np.random.choice(len(padded_mids))
		cmid=padded_mids[mid_index]
		if (len(cmid)-seq_length-1)<=0:
			padded_mids.pop(mid_index)
			continue
		seq_index=np.random.choice(len(cmid)-seq_length-1)
		batch.append(cmid[seq_index:seq_index+seq_length+1])
		if len(batch)==batch_size:
			yield batch
			batch=[]


batch_size=16
seq_length=512

from keras import callbacks as klc


tensorboard = klc.TensorBoard(
  log_dir='tensorboard',
  histogram_freq=0,
  batch_size=batch_size,
  write_graph=True,
  write_grads=True
)
tensorboard.set_model(model)


import time

t1=time.time()
tqdm_bar=tqdm()
for batch_id,batch in enumerate(generator(batch_size=batch_size,seq_length=seq_length)):
	batch=np.array(batch)
	loss=model.train_on_batch(batch[:,:-1],batch[:,1:])
	tensorboard.on_epoch_end(batch_id, {'loss':loss})
	if (time.time()-t1)>(60*5):
		t1=time.time()
		model.save("model.h5")
		tqdm_bar.update(1)


def add(batch,data_raw):
	sequence=batch[0]
	data_raw=r[0][-1]
	note=data_raw[:128+2]
	velocity=data_raw[128+2]
	time_=data_raw[128+1+2]
	data_=[]
	note_index=np.random.choice(range(len(note)),p=note)
	onehot=np.zeros(len(note))
	onehot[note_index]=1
	data_.extend(onehot)
	data_.append(velocity)
	data_.append(time_)
	sequence=sequence.tolist()
	sequence.append(data_)
	return np.array([sequence])

def predict_track(max_length):
	batch=np.zeros((1,1,130+2))
	batch[0][0][0]=1
	tqdm_bar=tqdm(total=max_length)
	for i in range(max_length):
		sequence=model.predict_on_batch(batch)
		tqdm_bar.update(1)
		if sequence[0][-1][1]==1:
			tqdm_bar.close()
			break
		batch=add(batch,sequence)
	return batch[0][1:]


def array2msg(sequence_array):
	res_track=mido.MidiTrack()
	res_track.append(mido.Message('program_change',program=0))
	for msg_array in sequence_array:
		notes=np.argmax(msg_array[2:128+2])
		vel=msg_array[128+2]
		time_n=msg_array[128+1+2]
		msg=mido.Message('note_on',note=int(notes),velocity=int(vel*127))
		msg.time=max(int(time_n),0)
		res_track.append(msg)
	res_track.append(mido.MetaMessage("end_of_track"))
	return res_track

sample_mid.tracks[1]=array2msg(predict_track(1000))
sample_mid.save("test.midi")
print("done saved")







