----------------------------------------------------------------

· AUDIO WAVEFORM
    - (1, T) 
    - Sampled at 16 kHz
    - Using "FFMPEG" 

· AUDIO EMBEDDINGS
    - (T, 256)
    - Sampled at 100 fps
    - Using "PASE+"

----------------------------------------------------------------

· FACES
    - (T, 128, 128, 3)
    - Sampled at 25 fps
    - Using "Retina Face Predictor"

· FACE EMBEDDING
    - TODO

· FACE LANDMARKS
    - (T, 68, 3)
    - Sampled at 25 fps
    - The 3rd coordinate of a landmark is the score/confidence
    - Using "FAN Predictor"    

----------------------------------------------------------------

· HAND LANDMARKS
    - (T, 2, 21, 5)
    - Sampled at 25 fps
    - The 4rd coordinate of a landmark is the visibility
    - The 5rd coordinate of a landmark is the presence

----------------------------------------------------------------

· POSE LANDMARKS
    - (T, 33, 5)
    - Sampled at 25 fps
    - The 4rd coordinate of a landmark is the visibility
    - The 5th coordinate of a landmark is the presence

----------------------------------------------------------------

