import numpy as np
import cv2

def draw_lines(img, lines):
    if len(lines.shape) > 2:
        lines = lines.squeeze()
    for x1,y1,x2,y2 in lines:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        
def average_length(lines):
    if len(lines.shape) > 2:
        lines = lines.squeeze()
    dx = lines[:,1] - lines[:,0]
    dy = lines[:,3] - lines[:,2]
    len_sq = dx*dx + dy*dy
    
    return np.mean(len_sq)


stripes = cv2.imread('stripes_input.png')
stripes_dist = cv2.imread('stripes_distorted.png')

edges_straight = cv2.Canny(stripes, 50, 150)
edges_dist = cv2.Canny(stripes_dist, 50, 150)


lines_straight = cv2.HoughLinesP(edges_straight,1,np.pi/180,200)
lines_dist = cv2.HoughLinesP(edges_dist,
                             rho=1,
                             theta=np.pi/180,
                             threshold=200)

lines_straight = lines_straight.squeeze()
lines_dist = lines_dist.squeeze()
  
print( average_length(lines_straight) )
print( average_length(lines_dist) )

draw_lines(stripes, lines_straight)
draw_lines(stripes_dist, lines_dist)

cv2.imshow('straight', stripes)
cv2.imshow('dist', stripes_dist)
cv2.waitKey(5000)
cv2.destroyAllWindows()