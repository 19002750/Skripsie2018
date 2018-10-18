from sympy.physics.vector import *
from sympy.matrices import *
import numpy as np
import math

from draw_qrcode import trueAdcmB


def triad2d(a,b):
    """a and b should be vectors containing the coordinates of the
        x,y and z-axis, all w.r.t the same reference frame

        a = [ax_x, ay_x, az_x]      b = [bx_x, by_x, bz_x]
            [ax_y, ay_y, az_y]          [bx_y, by_y, bz_y]
            [ax_z, ay_z, az_z]          [bx_z, by_z, bz_z]
    """
    #               Define origin reference frame
    O = ReferenceFrame('O')

    #               Extract the coordinates of each x,y and z-component of each x,y and z-axis
    ax_x = a[0][0]
    ax_y = a[1][0]
    ax_z = a[2][0]

    ay_x = a[0][1]
    ay_y = a[1][1]
    ay_z = a[2][1]

    az_x = a[0][2]
    az_y = a[1][2]
    az_z = a[2][2]

    bx_x = b[0][0]
    bx_y = b[1][0]
    bx_z = b[2][0]

    by_x = b[0][1]
    by_y = b[1][1]
    by_z = b[2][1]

    bz_x = b[0][2]
    bz_y = b[1][2]
    bz_z = b[2][2]

    #               Define a and b in this reference frame
    ax = ax_x*O.x + ax_y*O.y + ax_z*O.z
    ay = ay_x*O.x + ay_y*O.y + ay_z*O.z
    az = az_x*O.x + az_y*O.y + az_z*O.z

    bx = bx_x*O.x + bx_y*O.y + bx_z*O.z
    by = by_x*O.x + by_y*O.y + by_z*O.z
    bz = bz_x*O.x + bz_y*O.y + bz_z*O.z

    #               Now generate triad method matrixes
    AdcmO = Matrix([
        [dot(ax, O.x), dot(ax, O.y), dot(ax, O.z)],
        [dot(ay, O.x), dot(ay, O.y), dot(ay, O.z)],
        [dot(az, O.x), dot(az, O.y), dot(az, O.z)]
    ])

    BdcmO = Matrix([
        [dot(bx, O.x), dot(bx, O.y), dot(bx, O.z)],
        [dot(by, O.x), dot(by, O.y), dot(by, O.z)],
        [dot(bz, O.x), dot(bz, O.y), dot(bz, O.z)]
    ])

    BdcmA = AdcmO*BdcmO.transpose()
    AdcmB = BdcmO*AdcmO.transpose()

   #                Compute true values for testing
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    trueAdcmO = Matrix([
        [-0.5, -(np.sqrt(3)/2), 0],
        [(np.sqrt(3)/2), -0.5, 0],
        [0, 0, 1]
    ])

    trueBdcmO = Matrix([
        [-(np.sqrt(3)/2), -0.5, 0],
        [0.5, -(np.sqrt(3)/2), 0],
        [0, 0, 1]
    ])

    A.orient(O, 'DCM', trueAdcmO)
    B.orient(O, 'DCM', trueBdcmO)

    trueAdcmB = A.dcm(B)
    trueBdcmA = B.dcm(A)

    vectorA = 1*A.x + 1*A.y
    vectorB = 1*B.x + 1*B.y

    #               True Values
    truevectorA2B = trueBdcmA*(vectorA.to_matrix(A))
    truevectorB2A = trueAdcmB*(vectorB.to_matrix(B))

    truevectorA2B = truevectorA2B[0] * B.x + truevectorA2B[1] * B.y + truevectorA2B[2] * B.z
    truevectorB2A = truevectorB2A[0] * A.x + truevectorB2A[1] * A.y + truevectorB2A[2] * A.z

    #               Practical Values
    vectorA2B = BdcmA*(vectorA.to_matrix(A))
    vectorB2A = AdcmB*(vectorB.to_matrix(B))

    vectorA2B = vectorA2B[0] * B.x + vectorA2B[1] * B.y + vectorA2B[2] * B.z
    vectorB2A = vectorB2A[0] * A.x + vectorB2A[1] * A.y + vectorB2A[2] * A.z

    #               Get Error Matrix
    EdcmAB = AdcmB * trueAdcmB.transpose()
    EdcmBA = BdcmA * trueBdcmA.transpose()

    #               Get Error Angle
    truevectorA2B = truevectorA2B.normalize()
    truevectorB2A = truevectorB2A.normalize()
    vectorA2B = vectorA2B.normalize()
    vectorB2A = vectorB2A.normalize()

    Ea2b = dot(vectorA2B, truevectorA2B)
    Eb2a = dot(vectorB2A, truevectorB2A)
    print(Ea2b,Eb2a)
    EangleA2B = math.degrees(math.acos(dot(vectorA2B, truevectorA2B)))
    EangleB2A = math.degrees(math.acos(dot(vectorB2A, truevectorB2A)))



    #               Print Results
    print("trueAdcmB :", trueAdcmB)
    print("AdcmB:", AdcmB)
    print("trueBdcmA", trueBdcmA)
    print("BdcmA : ", BdcmA)
    print("truevectorA2B : ", truevectorA2B)
    print("truevectorB2A : ", truevectorB2A)
    print("vectorA2B : ", vectorA2B)
    print("vectorB2A : ", vectorB2A)
    print("Error matrix A2B : ", EdcmBA)
    print("Error angle A2B : ", EangleA2B)
    print("Error angle B2A : ", EangleB2A)


    return AdcmB,trueAdcmB






