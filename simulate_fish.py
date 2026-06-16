import numpy as np
import scipy.io
import time
import multiprocessing
from numba import jit
import numba
import sys
import os
from scipy.spatial import Delaunay
import scipy.linalg as linalg
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import paramiko
import threading
import os

# paramiko.util.log_to_file("paramiko_details.log", level="DEBUG")

host_name = "***"
port = 22
# username = "***"
# password = "***"
username = "***"
password = "***"
remote_path_root = "***"


def sftp_transfer(
    hostname, port, username, password, local_path, remote_path, status_dict
):
    # Initialize SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote server
        ssh.connect(hostname, port=port, username=username, password=password)

        # Open an SFTP session
        sftp = ssh.open_sftp()

        # Upload the file
        sftp.put(local_path, remote_path)
        print(f"File transferred to {remote_path}")
        status_dict["status"] = "Success"

        # Try to remove the local file if the transfer was successful
        os.remove(local_path)
        print(f"Local file {local_path} removed successfully.")

        # Close the SFTP session and SSH connection
        sftp.close()
        ssh.close()
    except Exception as e:
        print(f"Error: {e}")
        status_dict["status"] = "Failed"


def start_transfer_thread(hostname, port, username, password, local_path, remote_path):
    status_dict = {"status": "In Progress"}
    transfer_thread = threading.Thread(
        target=sftp_transfer,
        args=(hostname, port, username, password, local_path, remote_path, status_dict),
    )
    transfer_thread.start()
    return transfer_thread, status_dict


def create_remote_folder(hostname, port, username, password, remote_directory):
    # Initialize SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote server
    ssh.connect(hostname, port=port, username=username, password=password)
    # print(ssh)
    # Open an SFTP session
    sftp = ssh.open_sftp()
    sftp.get_channel().settimeout(10)
    # Try to create a directory
    print(sftp.chdir(remote_path_root))
    print(sftp.listdir(remote_path_root))
    try:
        sftp.chdir(remote_directory)  # Test if remote_path exists
    except IOError:
        sftp.mkdir(remote_directory)  # Create remote_path
        sftp.chdir(remote_directory)

        print(f"Directory {remote_directory} created successfully")
    # except IOError as e:
    #     print(f"Failed to create directory {remote_directory}: {e}")

    # Close the SFTP session and SSH connection
    sftp.close()
    ssh.close()


def list_files_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        return 0

    # List all files and directories in the folder
    files = os.listdir(folder_path)

    # Check if the list is empty (folder is empty)
    if not files:
        return 1
    else:
        # Return a list of files and directories
        return files


# Function for computation #
def randcircular(radius, num):
    r_pos = radius * np.sqrt(np.random.uniform(0, 1, num))
    theta = 2 * np.pi * np.random.uniform(0, 1, num)
    rx = r_pos * np.cos(theta)
    ry = r_pos * np.sin(theta)
    position = np.array((rx, ry))
    return position


def randractangle(width, height, num):
    rx = np.random.uniform(-width / 2, width / 2, num)
    ry = np.random.uniform(-height / 2, height / 2, num)
    position = np.array((rx, ry))
    return position


def GetInterc(p, R):
    a = np.cos(p[2])
    b = np.sin(p[2])
    x0 = p[0]
    y0 = p[1]
    t1 = (
        -2 * a * x0
        - 2 * b * y0
        + np.sqrt(
            (2 * a * x0 + 2 * b * y0) ** 2 - 4 * (a**2 + b**2) * (x0**2 + y0**2 - R**2)
        )
    ) / (2 * a**2 + 2 * b**2)
    sol = np.append(x0 + a * t1, y0 + b * t1)
    # intersection = sol
    # d = t1
    return sol, t1


def GetVoronoi(location, para):
    tri = Delaunay(np.transpose(location))
    v1 = np.ndarray.flatten(tri.simplices)
    v2 = np.ndarray.flatten(tri.simplices[:, [1, 2, 0]])
    vn = np.zeros((para["num"], para["num"]))
    vn[v1, v2] = 1
    vn = np.logical_or(vn, vn.T)
    return vn


def getangle(phi, rhox, rhoy, rhom=None):
    if rhom is None:
        rhom = np.sqrt(rhox**2 + rhoy**2)
    rhox = rhox / rhom
    rhoy = rhoy / rhom
    ex = np.cos(phi)
    ey = np.sin(phi)
    sgn = np.array(np.sign(ex * rhoy - ey * rhox))
    sgn[sgn == 0] = 1
    return sgn * np.arccos(np.clip(ex * rhox + ey * rhoy, -1, 1))


def GetVision(vn, state, para):
    listI = np.tile(np.arange(0, para["num"]), (para["num"]))
    ns = listI[np.ndarray.flatten(vn)]
    nn = np.sum(vn, axis=0)
    nnmax = np.amax(nn)

    neighborI = np.zeros((para["num"], nnmax + 1))
    neighborI[np.arange(0, para["num"]), nn] = para["num"]
    neighborI = np.cumsum(neighborI[:, :-1], axis=1).astype(int)
    neighborI[neighborI == 0] = ns
    x = state[0][:]
    y = state[1][:]
    a = state[2][:]
    xN = np.append(x, np.nan)
    yN = np.append(y, np.nan)
    aN = np.append(a, np.nan)
    phi = aN[neighborI] - a[:, None]
    rho1 = xN[neighborI] - x[:, None]
    rho2 = yN[neighborI] - y[:, None]
    rhon = np.sqrt(rho1**2 + rho2**2)

    # compute average distance between neighbors ignoring nan values
    average_distance = np.nanmean(rhon, axis=1)

    theta = getangle(a[:, None], rho1, rho2, rhon)
    wrap_theta = np.arctan2(np.sin(theta), np.cos(theta))

    # record all the theta and rhon pairs
    record_rhon_theta_pair = []
    for idx in range(para["num"]):
        record_rhon_theta_pair.extend(list(zip(rhon[idx, :], wrap_theta[idx, :])))

    visual = 1 + np.cos(theta)
    # visual_new = gaussian_func(wrap_theta, sigma)
    w_vision = np.nansum(
        (para["Ip"] * np.sin(phi) + rhon * np.sin(theta)) * visual,
        axis=1,
    ) / np.nansum(visual, axis=1)
    return w_vision


@numba.njit(parallel=True, fastmath=True)
def GetHydro_numba(state, num, If):
    Uc = np.zeros(num, dtype=np.complex128)
    wc = np.zeros(num, dtype=np.float64)
    for i in numba.prange(num):
        for j in range(num):
            if i != j:
                dZr = state[0][i] - state[0][j]
                dZi = state[1][i] - state[1][j]
                dZ = dZr + 1j * dZi
                o_di = state[2][j]
                ori = state[2][i]
                Uc[i] += np.exp(1j * o_di) / (dZ**2) * If / np.pi
                wc[i] += (
                    np.imag(np.exp(1j * (2 * ori + o_di)) / (dZ**3)) * 2 * If / np.pi
                )
    U = np.zeros((2, num), dtype=np.float64)
    U[0, :] = np.real(Uc)
    U[1, :] = -np.imag(Uc)
    return U, wc


@numba.njit(parallel=True, fastmath=True)
def GetHydro_reg_numba(state, num, If, delta):
    Uc = np.zeros(num, dtype=np.complex128)
    wc = np.zeros(num, dtype=np.float64)
    for i in numba.prange(num):
        for j in range(num):
            if i != j:
                dZr = state[0][i] - state[0][j]
                dZi = state[1][i] - state[1][j]
                dZ = dZr + 1j * dZi
                o_di = state[2][j]
                ori = state[2][i]
                Uc[i] += np.exp(1j * o_di) / (dZ**2 + delta**2) * If / np.pi
                wc[i] += (
                    np.imag(np.exp(1j * (2 * ori + o_di)) * dZ / (dZ**2 + delta**2))
                    * 2
                    * If
                    / np.pi
                )
    U = np.zeros((2, num), dtype=np.float64)
    U[0, :] = np.real(Uc)
    U[1, :] = -np.imag(Uc)
    return U, wc


def GetHydro(state, para):
    # generate index mat
    otherI = np.tile(np.arange(0, para["num"]), (para["num"], 1))
    otherI = np.delete(otherI, np.arange(0, otherI.size, para["num"] + 1)).reshape(
        (para["num"], para["num"] - 1)
    )
    ori = state[2][:]
    dZr = state[0][:][:, None] - state[0][:][otherI]
    dZi = state[1][:][:, None] - state[1][:][otherI]
    dZ = dZr + 1j * dZi
    o_di = ori[otherI]
    Uc = np.sum((np.exp(1j * o_di) / (dZ**2)), axis=1) * para["If"] / np.pi
    wc = (
        np.sum(
            np.imag(np.exp(1j * (2 * ori[:, None] + o_di)) / (dZ**3)),
            axis=1,
        )
        * 2
        * para["If"]
        / np.pi
    )
    Ux = np.real(Uc)
    Uy = -np.imag(Uc)
    w = wc
    U = np.row_stack([Ux, Uy])
    return U, w


def GetNoise(para):
    wNoise = np.zeros([3, para["num"]])
    wNoise[2][:] = (
        para["In"] * np.sqrt(para["dt"]) * np.random.normal(0, 1, para["num"])
    )
    return wNoise


def Initialization_segment(para):
    # load npy file
    B1_init = np.load("/Users/chenchen/Dropbox/B_letter_points.npy")
    O_init = np.load("/Users/chenchen/Dropbox/O_letter_points.npy")
    B2_init = np.load("/Users/chenchen/Dropbox/B_letter_points.npy")
    O_init = O_init.T / 10
    B1_init = B1_init.T / 10
    B2_init = B2_init.T / 10
    O_init[0, :] = O_init[0, :] + 30
    B2_init[0, :] = B2_init[0, :] + 60
    O_init[1, :] = -O_init[1, :]
    B1_init[1, :] = -B1_init[1, :]
    B2_init[1, :] = -B2_init[1, :]
    O_init[1, :] = O_init[1, :] + 20
    B1_init[1, :] = B1_init[1, :] + 20
    B2_init[1, :] = B2_init[1, :] + 20
    # concatenate all the points
    all_points = np.concatenate((B1_init, O_init, B2_init), axis=1)

    plt.scatter(all_points[0, :], all_points[1, :])
    plt.show()
    total_steps = int(para["total_time"] / para["dt"] + 1)
    saving_steps = int(para["saving_window"] / para["dt"])
    state = np.zeros((3, para["num"], saving_steps + 1))
    rDotList = np.zeros((2, para["num"], saving_steps + 1))

    alpha = np.random.uniform(-np.pi, np.pi, para["num"])
    # r = randractangle(para["R"], para["R"], para["num"])
    r = all_points
    count = 0
    state[:, :, count] = np.row_stack([r, alpha])
    rDotList[:, :, count] = np.ones((2, para["num"]))
    # wNoiseList[:, :, count] = np.zeros((3, para['num']))
    return state, rDotList, count, total_steps


def Initialization(para):

    total_steps = int(para["total_time"] / para["dt"] + 1)
    saving_steps = int(para["saving_window"] / para["dt"])
    state = np.zeros((3, para["num"], saving_steps + 1))
    rDotList = np.zeros((2, para["num"], saving_steps + 1))
    wVisionList = np.zeros((para["num"], saving_steps + 1))  # ← 新增

    alpha = np.random.uniform(-np.pi, np.pi, para["num"])
    r = randcircular(para["R"], para["num"])
    count = 0
    state[:, :, count] = np.row_stack([r, alpha])
    rDotList[:, :, count] = np.ones((2, para["num"]))
    wVisionList[:, count] = 0.0

    return state, rDotList, wVisionList, count, total_steps


def reInitialization(para, state, rDotList, wVisionList):
    previous_state = state[:, :, -1]
    previous_rdot = rDotList[:, :, -1]
    previous_wvision = wVisionList[:, -1]

    saving_steps = int(para["saving_window"] / para["dt"])
    state = np.zeros((3, para["num"], saving_steps + 1))
    rDotList = np.zeros((2, para["num"], saving_steps + 1))
    wVisionList = np.zeros((para["num"], saving_steps + 1))

    count = 0
    state[:, :, count] = previous_state
    rDotList[:, :, count] = previous_rdot
    wVisionList[:, count] = previous_wvision

    return state, rDotList, wVisionList, count


def step_Computation(para, restart_index=0):
    saving_steps = int(para["saving_window"] / para["dt"])

    [state, rDotList, wVisionList, count, steps] = Initialization(para)

    file_counter = 0
    if restart_index > 0:
        file_counter = restart_index

        loaded_data = scipy.io.loadmat(
            para["local_folder_name"] + "{:}.mat".format(file_counter)
        )

        state = loaded_data["state"]
        rDotList = loaded_data["rdot"]
        wVisionList = loaded_data["wvision"]

        file_counter += 1
        [state, rDotList, count] = reInitialization(para, state, rDotList, wVisionList)
    for step in tqdm(range(restart_index * saving_steps, steps)):

        CurState = state[:, :, count]
        CurLocation = state[0:2, :, count]

        CurHeading = state[2, :, count]
        ex = np.cos(CurHeading)
        ey = np.sin(CurHeading)
        e = np.row_stack([ex, ey])

        # [U, wHydro] = GetHydro(CurState, para)
        [U, wHydro] = GetHydro_numba(CurState, para["num"], para["If"])
        # [U, wHydro] = GetHydro_reg_numba(
        #     CurState, para["num"], para["If"], para["delta"]
        # )

        voro = GetVoronoi(CurLocation, para)

        wVision = GetVision(
            voro,
            CurState,
            para,
        )
        wNoise = GetNoise(para)
        # wWall = GetWallAvoid(CurState, para)

        thetaDot = wVision  # + wHydro  ← 只剩视觉交互
        rDot = e # + U       ← 不加流体速度

        count += 1
        state[:, :, count] = (
            CurState + np.row_stack([rDot, thetaDot]) * para["dt"] + wNoise
        )

        rDotList[:, :, count] = rDot
        wVisionList[:, count] = wVision

        if count == saving_steps:
            savestate = state[:, :, :]
            saverDotList = rDotList[:, :, :]
            savewVisionList = wVisionList[:, :]
            scipy.io.savemat(
                para["local_folder_name"] + "{:}.mat".format(file_counter),
                mdict={"state": savestate, "rdot": saverDotList, "wvision": savewVisionList,},
            )

            file_counter += 1
            [state, rDotList, wVisionList, count] = reInitialization(para, state, rDotList, wVisionList)

    return state, rDotList, wVisionList


def Computation(n, parameters=[1.5, 0.3, 100, 10000, 1000], restart_index=0):
    # np.random.seed(n + int(time.time()))
    np.random.seed(n)

    para = dict(
        [
            ("Ip", parameters[0]),
            ("In", parameters[1]),
            ("R", parameters[2]),
            ("light", 1),
            ("dt", 1e-2),
            ("Iw", 0.94),
            ("If", 0),
            ("num", parameters[3]),
            ("delta", 1e-2),
            ("total_time", parameters[4]),
            ("saving_window", 1),
        ]
    )

    # current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    root = "./"
    folder_name = (
        "Ip{:02d}_In{:02d}_R{:}_time{:}_num{:}_{}".format(
            int(para["Ip"] * 10),
            int(para["In"] * 10),
            int(para["R"]),
            int(para["total_time"]),
            int(para["num"]),
            n,
        )
        + "/"
    )
    if not os.path.exists(root + folder_name):
        os.makedirs(root + folder_name)
    para["local_folder_name"] = root + folder_name
    para["remote_folder_name"] = remote_path_root + folder_name

    sys.stdout.write(
        "Ip = %f; In = %f; R = %d; Light = %f; Number = %d\n"
        % (para["Ip"], para["In"], para["R"], para["light"], para["num"])
    )
    step_Computation(para, restart_index)
    return None


##########################################################
if __name__ == "__main__":

    parameters_list = [
        [9, 0.5, 10, 100, 100],
    ]

    for index, parameters in enumerate(parameters_list):
        # print(parameters)
        Computation(0, parameters)