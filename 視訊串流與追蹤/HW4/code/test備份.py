import numpy as np
import cv2
import ffmpy
import ffmpeg
import subprocess
writer = None
cap = cv2.VideoCapture(0)
frameIndex = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    #cv2.imwrite("output/frame-{}.jpg".format(frameIndex), frame)
    print(frameIndex)
    if frameIndex > 1000:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("input.avi", fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        
        #if total > 0:
        #    elap = (end - start)
            #print("[INFO] single frame took {:.4f} seconds".format(elap))
            #print("[INFO] estimated total time to finish: {:.4f}".format(
            #    elap * total))

    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex = frameIndex + 1
    
    if frameIndex % 100 == 0:
        stream = ffmpeg.input('input.avi')
        stream = ffmpeg.output(stream, 'output.mp4')
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream)
        subprocess.call(["MP4Box","-dash", "10", "output.mp4"],shell=False)
#ff = ffmpy.FFmpeg(
#     inputs={'input.avi': None},
#     outputs={'output.mp4': None})
#ff.run()





# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()