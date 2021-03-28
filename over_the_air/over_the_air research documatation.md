Adversarial-Robustness-Toolbox Research Documentation

**Kinetics 400 Exploration**
1. File Formats: Three csv files and three json files: test.csv, train csv, and validate.csv (all the video comes from youtube). There is a youtube_id category that we can get all the video
2. | label | youtube_id | time_start | time_end | split | example link |
   | -- | -- | -- | -- | -- | -- |
   | drinking beer | --6bJUbfpnQ | 17 | 27 | test | https://youtu.be/--6bJUbfpnQ?t=17 | 
3. Our traing data that located in GIthub over_the_air branch: test.csv, train.csv and validate.csv
**(I3D Exploration)**
1. The over the air research paper link to the https://github.com/google/mediapipe
2. The video inputs to the model need to have the same dimensions batch_size x num_frames x 224 x 224 x num_channels
3. Blockers:  I3D can only support python 3.6 libraries and also Cuda versions
4. SOLVED by using new models mmaction2 https://github.com/open-mmlab/mmaction2 works well on Kinetic 400
**Mideapipe**
1. Preprocessing the video
2. Blockers: only support C++ for video processing
3. SOLVED: After discussion with Mr.Beater from IBM(The head of this Open Source project) we don’t need to preprocess the video
