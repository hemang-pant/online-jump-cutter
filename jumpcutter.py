
import subprocess
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import copyfile, rmtree
import os
import argparse
from pytube import YouTube
#made with love :)
def downloadFile(url):
    name = YouTube(url).streams.first().download()
    newname = name.replace(' ','_')
    os.rename(name,newname)
    return newname

def getMaxVolume(s): #returns max volume of wav file, input is an array
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

def copyFrame(inputFrame,outputFrame):  #copies frame from input to output
    src = TEMP_FOLDER+"/frame{:06d}".format(inputFrame+1)+".jpg"
    dst = TEMP_FOLDER+"/newFrame{:06d}".format(outputFrame+1)+".jpg"
    if not os.path.isfile(src):
        return False
    copyfile(src, dst) 
    if outputFrame%20 == 19:
        print(str(outputFrame+1)+" time-altered frames saved.")
    return True

def inputToOutputFilename(filename): #when user just inputs a filename, this function converts it to the output filename
    dotIndex = filename.rfind(".")
    return filename[:dotIndex]+"_ALTERED"+filename[dotIndex:]

def createPath(s): #creates path if it doesn't exist
    try:  
        os.mkdir(s)
    except OSError:  
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"

def deletePath(s): #deletes path if it exists
    try:  
        rmtree(s,ignore_errors=False)
    except OSError:  
        print ("Deletion of the directory %s failed" % s)
        print(OSError)

parser = argparse.ArgumentParser(description='Modifies a video file to play at different speeds when there is sound vs. silence.') #creates parser
parser.add_argument('--input_file', type=str,  help='the video file you want modified') #adds input file
parser.add_argument('--url', type=str, help='A youtube url to download and process') #adds url
parser.add_argument('--output_file', type=str, default="", help="the output file. (optional. if not included, it'll just modify the input file name)") #adds output file
parser.add_argument('--silent_threshold', type=float, default=0.03, help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". It ranges from 0 (silence) to 1 (max volume)") #adds silent threshold
parser.add_argument('--sounded_speed', type=float, default=1.00, help="the speed that sounded (spoken) frames should be played at. Typically 1.") #adds sounding speed
parser.add_argument('--silent_speed', type=float, default=5.00, help="the speed that silent frames should be played at. 999999 for jumpcutting.") #adds silent speed
parser.add_argument('--frame_margin', type=float, default=1, help="some silent frames adjacent to sounded frames are included to provide context. How many frames on either the side of speech should be included? That's this variable.") #adds frame margin
parser.add_argument('--sample_rate', type=float, default=44100, help="sample rate of the input and output videos") 
parser.add_argument('--frame_rate', type=float, default=30, help="frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.")
parser.add_argument('--frame_quality', type=int, default=3, help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 is the default.")

args = parser.parse_args()



frameRate = args.frame_rate
SAMPLE_RATE = args.sample_rate
SILENT_THRESHOLD = args.silent_threshold
FRAME_SPREADAGE = args.frame_margin
NEW_SPEED = [args.silent_speed, args.sounded_speed]
if args.url != None:
    INPUT_FILE = downloadFile(args.url)
else:
    INPUT_FILE = args.input_file
FRAME_QUALITY = args.frame_quality

assert INPUT_FILE != None , "You forgot to put the input file :("

if len(args.output_file) >= 1:
    OUTPUT_FILE = args.output_file
else:
    OUTPUT_FILE = inputToOutputFilename(INPUT_FILE)

TEMP_FOLDER = "TEMP"
AUDIO_FADE_ENVELOPE_SIZE = 400
createPath(TEMP_FOLDER)

command = "ffmpeg -i "+INPUT_FILE+" -qscale:v "+str(FRAME_QUALITY)+" "+TEMP_FOLDER+"/frame%06d.jpg -hide_banner" #converts frames to .jpg
subprocess.call(command, shell=True)

command = "ffmpeg -i "+INPUT_FILE+" -ab 160k -ac 2 -ar "+str(SAMPLE_RATE)+" -vn "+TEMP_FOLDER+"/audio.wav"  #converts audio to .wav

subprocess.call(command, shell=True)




sampleRate, audioData = wavfile.read(TEMP_FOLDER+"/audio.wav") #reads audio file
audioSampleCount = audioData.shape[0] #number of samples in audio file
maxAudioVolume = getMaxVolume(audioData) #max volume of audio file

samplesPerFrame = sampleRate/frameRate #how many samples per frame

audioFrameCount = int(math.ceil(audioSampleCount/samplesPerFrame)) #how many frames to extract from audio file

hasLoudAudio = np.zeros((audioFrameCount)) #array to keep track of which frames have loud audio



for i in range(audioFrameCount):
    start = int(i*samplesPerFrame) #start of current frame
    end = min(int((i+1)*samplesPerFrame),audioSampleCount) #end of current frame
    audiochunks = audioData[start:end] #samples of current frame
    maxchunksVolume = float(getMaxVolume(audiochunks))/maxAudioVolume #max volume of samples in current frame
    if maxchunksVolume >= SILENT_THRESHOLD: 
        hasLoudAudio[i] = 1 #input value in has loud audio

chunks = [[0,0,0]] #array to keep track of chunks and their start and end times
shouldIncludeFrame = np.zeros((audioFrameCount)) #array to keep track of which frames should be included
for i in range(audioFrameCount): 
    start = int(max(0,i-FRAME_SPREADAGE)) #start of current chunk
    end = int(min(audioFrameCount,i+1+FRAME_SPREADAGE)) #end of current chunk
    shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end]) #whether this chunk should be included
    if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]):  
        chunks.append([chunks[-1][1],i,shouldIncludeFrame[i-1]]) #append chunk to chunks

chunks.append([chunks[-1][1],audioFrameCount,shouldIncludeFrame[i-1]]) #append chunk to chunks
chunks = chunks[1:] #remove first chunk (start time = 0)

outputAudioData = np.zeros((0,audioData.shape[1])) #array to keep track of audio data to be written to output file
outputPointer = 0 #pointer to keep track of where to write audio data to output file

lastExistingFrame = None #last existing frame
for chunk in chunks: 
    audioChunk = audioData[int(chunk[0]*samplesPerFrame):int(chunk[1]*samplesPerFrame)] #samples of current chunk
    
    sFile = TEMP_FOLDER+"/tempStart.wav" #temp file to be used for start of current chunk
    eFile = TEMP_FOLDER+"/tempEnd.wav" #temp file to be used for end of current chunk
    wavfile.write(sFile,SAMPLE_RATE,audioChunk) #writes start of current chunk to temp file
    with WavReader(sFile) as reader: #reads start of current chunk
        with WavWriter(eFile, reader.channels, reader.samplerate) as writer: #writes end of current chunk
            tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])]) #creates phase vocoder object
            tsm.run(reader, writer) #runs phase vocoder
    _, alteredAudioData = wavfile.read(eFile) #reads end of current chunk
    leng = alteredAudioData.shape[0] #length of altered audio data
    endPointer = outputPointer+leng #end pointer value upadted to be the end of the current chunk
    outputAudioData = np.concatenate((outputAudioData,alteredAudioData/maxAudioVolume)) #adds altered audio data to output audio data

    
    if leng < AUDIO_FADE_ENVELOPE_SIZE: #if the length of the altered audio data is less than the fade envelope size
        outputAudioData[outputPointer:endPointer] = 0 #set the values in the output audio data to 0
    else:
        premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE)/AUDIO_FADE_ENVELOPE_SIZE #creates a mask to fade the audio data
        mask = np.repeat(premask[:, np.newaxis],2,axis=1) # make the fade-envelope mask stereo todo : make this work with more than 2 channels
        outputAudioData[outputPointer:outputPointer+AUDIO_FADE_ENVELOPE_SIZE] *= mask #fades the audio data
        outputAudioData[endPointer-AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1-mask #fades the audio data

    startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame)) #start of current output frame
    endOutputFrame = int(math.ceil(endPointer/samplesPerFrame)) #end of current output frame
    for outputFrame in range(startOutputFrame, endOutputFrame): 
        inputFrame = int(chunk[0]+NEW_SPEED[int(chunk[2])]*(outputFrame-startOutputFrame)) #input frame of current output frame
        didItWork = copyFrame(inputFrame,outputFrame) #copy frame from input to output
        if didItWork: 
            lastExistingFrame = inputFrame #it worked :)
        else: 
            copyFrame(lastExistingFrame,outputFrame) #it will work anyhow

    outputPointer = endPointer #updates output pointer

wavfile.write(TEMP_FOLDER+"/audioNew.wav",SAMPLE_RATE,outputAudioData) #writes output audio data to output file
command = "ffmpeg -framerate "+str(frameRate)+" -i "+TEMP_FOLDER+"/newFrame%06d.jpg -i "+TEMP_FOLDER+"/audioNew.wav -strict -2 "+OUTPUT_FILE #command to convert frames to video
subprocess.call(command, shell=True) #runs command

deletePath(TEMP_FOLDER)

