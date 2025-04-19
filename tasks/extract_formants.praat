# 📁 Input: path to your .wav file

form Get file I/O data
    sentence: "Sound path", "/Users/ben/iwonder.wav"
    sentence: "Output path", "/Users/ben/iwonder_formants.csv"
endform

# 🧠 Parameters
maxFormant = 5500
numFormants = 5
windowLength = 0.025
timeStep = 0.005

# 🔊 Read sound
Read from file... 'sound_path$'
soundName$ = selected$("Sound")

# 📈 Analyze formants
To Formant (burg)... 0 'numFormants' 'maxFormant' 'windowLength' 50
formantName$ = selected$("Formant")

# 📝 Write header
writeFileLine: "'outputPath$'", "time,F1,F2,F3,F4,F5"

# 📊 Loop over time
duration = Get total duration
writeInfoLine: "Duration: ", duration, " seconds"
t = 0
while t < duration

    f1 = Get value at time... 1 t Hertz Linear
    f2 = Get value at time... 2 t Hertz Linear
    f3 = Get value at time... 3 t Hertz Linear
    f4 = Get value at time... 4 t Hertz Linear
    f5 = Get value at time... 5 t Hertz Linear
    
    
    # Only write if at least F1 exists
    if f1 != undefined
        appendFileLine: "'output_path$'", t, ",", f1, ",", f2, ",", f3, ",", f4, ",", f5
    endif

    t = t + 'timeStep'
endwhile

writeInfoLine: "✅ Done! Formants saved to: ", output_path$
