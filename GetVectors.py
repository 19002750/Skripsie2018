import numpy as np


def get_coords(A, B):
    """A and B must contain the coordinates of origin , x-axis,y-axis and z-axis
        both with respect to the same reference frame.

        A = [origin     ]
            [x = (x,y,z)]
            [y = (x,y,z)]
            [z = (x,y,z)]
                            ,each having x,y and z-components
        This function extract the values to generate x,y and z coordinate base vectors
         of A and B respectively with respect to the same reference frame

         Each point, (origin,x,y,z), of A and B must be defined in the same reference frame
         i.o.w the all the measurements must be with respect to the same coordinate system.
    """
    #               Get A's origin,x,y and z points
    ao_coords = A[0]
    ax_coords = A[1]
    ay_coords = A[2]
    az_coords = A[3]

    #               Then subdivide coordinates into there x,y and z components
    #               Origin
    ao_x = ao_coords[0]
    ao_y = ao_coords[1]
    ao_z = ao_coords[2]
    #               X_AXIS
    ax_x = ax_coords[0]
    ax_y = ax_coords[1]
    ax_z = ax_coords[2]
    #               Y_AXIS
    ay_x = ay_coords[0]
    ay_y = ay_coords[1]
    ay_z = ay_coords[2]
    #               Z-AXIS
    az_x = az_coords[0]
    az_y = az_coords[1]
    az_z = az_coords[2]

    #               Get B's origin,x,y and z points
    bo_coords = B[0]
    bx_coords = B[1]
    by_coords = B[2]
    bz_coords = B[3]
    #               Then subdivide each into there x,y and z components
    #               Origin
    bo_x = bo_coords[0]
    bo_y = bo_coords[1]
    bo_z = bo_coords[2]
    #               X_AXIS
    bx_x = bx_coords[0]
    bx_y = bx_coords[1]
    bx_z = bx_coords[2]
    #               Y_AXIS
    by_x = by_coords[0]
    by_y = by_coords[1]
    by_z = by_coords[2]
    #               Z_AXIS
    bz_x = bz_coords[0]
    bz_y = bz_coords[1]
    bz_z = bz_coords[2]

    #               Construct each axis's matrix, where each axis has x,y and z-components
    ax = np.array([
        [ax_x - ao_x],
        [ax_y - ao_y],
        [ax_z - ao_z]
    ])
    axu = normilize(ax)

    ay = np.array([
        [ay_x - ao_x],
        [ay_y - ao_y],
        [ay_z - ao_z]
    ])
    ayu = normilize(ay)

    az = np.array([
        [az_x - ao_x],
        [az_y - ao_y],
        [az_z - ao_z]
    ])
    azu = normilize(az)

    bx = np.array([
        [bx_x - bo_x],
        [bx_y - bo_y],
        [bx_z - bo_z]
    ])
    bxu = normilize(bx)

    by = np.array([
        [by_x - bo_x],
        [by_y - bo_y],
        [by_z - bo_z]
    ])
    byu = normilize(by)

    bz = np.array([
        [bz_x - bo_x],
        [bz_y - bo_y],
        [bz_z - bo_z]
    ])
    bzu = normilize(bz)

    #               Create the Reference frame base vectors

    a = np.column_stack((axu, ayu, azu))
    b = np.column_stack((bxu, byu, bzu))

    return a, b

def normilize(v):
    vx = v[0]
    vy = v[1]
    vz = v[2]

    vu = v/(np.sqrt((vx*vx + vy*vy + vz*vz)))
    return vu
