import numpy as np

# Color space can impact results,
# LUV seems more accurate for computer vision
# "LUV decouple the "color" (chromaticity, the UV part) and "lightness" (luminance, the L part) of color. Thus in object detection, it is common to match objects just based on the UV part, which gives invariance to changes in lighting condition."
# function to change color space
# from RGB to LUV
# src: http://ilkeratalay.com/colorspacesfaq.php#spaceconv
# CCIR D65
XYZ = [[0.430574, 0.341550, 0.178325], [
    0.222015, 0.706655, 0.071330], [0.020183, 0.129553, 0.939180]]
RGB = [[3.063, -1.393, -0.476], [-0.969, 1.876, 0.042], [0.068, -0.229, 1.069]]
# coordinate of the white point
xn = 0.312713
yn = 0.329016
zn = 1-(xn+yn)
Xn = xn/yn
Yn = 1
Zn = zn/yn
Lt = 0.008856
# same definitions for u' and v'
# but applied to the white point reference
u_n_prime = 4*Xn/(Xn+15*Yn*+3*Zn)
v_n_prime = 9*Yn/(Xn+15*Yn+3*Zn)


def RGBtoLUV(RGB_value):
    LUV_value = np.empty(3)
    # RGB to XYZ
    r = RGB_value[0]
    g = RGB_value[1]
    b = RGB_value[2]

    x = XYZ[0][0]*r+XYZ[0][1]*g+XYZ[0][2]*b
    y = XYZ[1][0]*r+XYZ[1][1]*g+XYZ[1][2]*b
    z = XYZ[2][0]*r+XYZ[2][1]*g+XYZ[2][2]*b

    # XYZ to LUV
    # Compute Luminancy: L*
    L0 = y/(255.*Yn)
    if(L0 > Lt):
        LUV_value[0] = 116. * pow(L0, 1/3.) - 16.
    else:
        LUV_value[0] = 903.3 * L0
    # Chrominancy: u' & v'
    c = x + 15 * y + 3 * z
    if(c != 0):
        u_prime = (4*x)/c
        v_prime = (9*y)/c
    else:
        u_prime = 4.
        v_prime = 9./15.
    # u'n & v'n
    LUV_value[1] = (13 * LUV_value[0] * (u_prime-u_n_prime))
    LUV_value[2] = (13 * LUV_value[0] * (v_prime-v_n_prime))
    # print(LUV_value)
    return LUV_value


def boundary(value, min, max):
    if(value < min):
        value = min
    if(value > max):
        value = max
    return value

# from LUV to RGB


def LUVtoRGB(LUV_value):
    rgb = np.empty(3)
    L = LUV_value[0]
    u = LUV_value[1]
    v = LUV_value[2]

    if(L < 0.1):
        print(L)
        rgb = np.zeros(3)
    else:
        # LUV to XYZ
        if(L < 8.):
            y = Yn*L/903.3
        else:
            y = (L+16.)/116.
            y = Yn*(y**3)
        u_prime = u/(13*L) + u_n_prime
        v_prime = v/(13*L) + v_n_prime

        x = 9 * u_prime * y/(4*v_prime)
        z = (12 - 3*u_prime - 20 * v_prime)*y / (4 * v_prime)

        # XYZ to RGB
        r = round((RGB[0][0]*x+RGB[0][1]*y+RGB[0][2]*z)*255.)
        g = round((RGB[1][0]*x+RGB[1][1]*y+RGB[1][2]*z)*255.)
        b = round((RGB[2][0]*x+RGB[2][1]*y+RGB[2][2]*z)*255.)

        rgb[0] = boundary(r, 0., 255.)
        rgb[1] = boundary(g, 0., 255.)
        rgb[2] = boundary(b, 0., 255.)

    return rgb
