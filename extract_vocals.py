import compiam
path = "thodi/Koluvamaregatha/Koluvamaregatha.multitrack-vocal-alaapana.mp3"
ftanet_carnatic = compiam.load_model("melody:ftanet-carnatic")
ftanet_pitch_track = ftanet_carnatic.predict(path, hop_size=30)