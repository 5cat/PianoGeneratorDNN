import numpy as np
from keras.models import load_model
import mido
from tqdm import tqdm


model=load_model("model.h5",compile=False)

def temp_f(output,temp):
	output = np.log(output + 1e-8) / temp
	output = np.exp(output)
	return output / np.sum(output)

def add(batch,data_raw,temp_value):
	sequence=batch[0]
	data_raw=data_raw[0][-1]
	note=data_raw[:128+2]
	velocity=data_raw[128+2]
	time_=data_raw[128+1+2]
	data_=[]
	note_index=np.random.choice(range(len(note)),p=temp_f(note,temp_value))
	onehot=np.zeros(len(note))
	onehot[note_index]=1
	data_.extend(onehot)
	data_.append(velocity)
	data_.append(time_)
	sequence=sequence.tolist()
	sequence.append(data_)
	return np.array([sequence])

def predict_track(max_length,temp_value):
	batch=np.zeros((1,1,130+2))
	batch[0][0][0]=1
	tqdm_bar=tqdm(total=max_length)
	for i in range(max_length):
		sequence=model.predict_on_batch(batch)
		tqdm_bar.update(1)
		if sequence[0][-1][1]==1:
			tqdm_bar.close()
			break
		batch=add(batch,sequence,temp_value)
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


sample_mid=mido.MidiFile("maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi")

while True:
	print()
	print("number of wanted messages = ",end='')
	n_msgs=int(input())
	print("the temp value = ",end='')
	temp_value=float(input())
	print("midi file name = ",end='')
	midi_file_name=str(input())+'.midi'
	sample_mid.tracks[1]=array2msg(predict_track(n_msgs,temp_value))
	sample_mid.save(midi_file_name)
	print("do you want generate another sequence? [Y/N]")
	resp=str(input()).lower()
	if resp=='n':
		break