Data_Retriever.py
from mido import MidiFile
 
midi = MidiFile('/Users/PabloLoo1/Documents/Texas A&M University/Spring 2021/'
               + 'BMEN 489/Project 2/'
               + 'score.mid',
               #+ 'WA_Mozart_Marche_Turque_Turkish_March_fingered.mid',
               clip=True)
 
midiString = ''
 
# There are two tracks, which correspont to each hand
# playing the piano
 
# 2 tracks, so midi.tracks[0] or midi.tracks[1]
 
 
for i in range(0, len(midi.tracks[1])):
   # print(midi.tracks[0][i])
   midiString += str(midi.tracks[1][i]) + '\n'
 
print('\n')
print('Input:')
print(midiString[:3000])
