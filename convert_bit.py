import numpy
import cv2

def convert_bit (output_bits_per_color, color_image, input_bits_per_color = 8):
	return numpy.apply_along_axis(
		reduce_bgr, #helper function
		2, #dimension for function to be applied
		color_image,
		output_bits_per_color,
		input_bits_per_color
	)

# takes an array representation of color in bgr form  with max values of 255
# and scales it down to the give the array scaled down where the max value is
# 2^(bits/3) - 1 this function is called by convert_bit
def reduce_bgr (bgr_array, output_bits_per_color, input_bits_per_color):
	max_color_value = 2 ** output_bits_per_color - 1
	input_max_color_value = 2 ** input_bits_per_color - 1
	new_bgr = [numpy.uint8(round(x * max_color_value / input_max_color_value)) for x in bgr_array]
	return new_bgr
