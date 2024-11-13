import os
import time
import numpy as np
import itertools
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def extract_features(H, n_components=50):
    """
    Extract spatial and frequency features from the complex channel matrix H using Complex PCA.
    
    Parameters:
        H (numpy.ndarray): Complex channel matrix with shape (samples, 64, 2, 408).
        n_components (int): Number of dimensions to reduce to using PCA.
        
    Returns:
        numpy.ndarray: Reduced feature representation (samples, n_components).
    """
    num_samples, num_antennas, num_user_antennas, num_subcarriers = H.shape

    # Step 1: Compute spatial covariance matrix
    spatial_features = []
    for sample in H:
        # Combine user antenna dimensions into a single matrix (64, 816)
        reshaped_sample = sample.reshape(num_antennas, -1)  # Shape: (64, 816)
        # Compute spatial covariance (64x64)
        spatial_cov = np.matmul(reshaped_sample, reshaped_sample.conj().T)
        spatial_features.append(spatial_cov.flatten())  # Flatten into 1D array

    spatial_features = np.array(spatial_features)  # Shape: (samples, 64*64)

    # Step 2: Compute frequency domain features
    frequency_features = []
    for sample in H:
        # Average across antennas and user antennas for each subcarrier (408)
        freq_avg = np.mean(np.abs(sample), axis=(0, 1))  # Shape: (408,)
        frequency_features.append(freq_avg)

    frequency_features = np.array(frequency_features)  # Shape: (samples, 408)

    # Step 3: Combine spatial and frequency features
    combined_features = np.hstack((spatial_features, frequency_features))  # Shape: (samples, 64*64 + 408)

    # Step 4: Apply Complex PCA for dimensionality reduction
    # Compute covariance matrix of the combined features
    cov_matrix = np.cov(combined_features.T)  # Shape: (features, features)
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # Sort by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    # Select the top n_components eigenvectors
    principal_components = eigenvectors[:, :n_components]  # Shape: (features, n_components)
    # Project data onto the principal components
    reduced_features = np.dot(combined_features, principal_components)  # Shape: (samples, n_components)

    return reduced_features

def convert_to_real_amplitude_phase(H):
    """
    Convert complex data to a real-valued matrix using amplitude and phase.
    
    Parameters:
        H (numpy.ndarray): Complex-valued matrix with shape (samples, features).
        
    Returns:
        numpy.ndarray: Real-valued matrix with shape (samples, features * 2).
    """
    amplitude = np.abs(H) 
    phase = np.angle(H)   
    return np.hstack((amplitude, phase))

def reduce_dimension_tsne(features, n_components=2, batch_size=1000):
    """
    使用 t-SNE 将高维数据降维到 2D。
    
    Parameters:
        features (numpy.ndarray): 高维特征矩阵，形状为 (samples, n_features)。
        n_components (int): 降维后的目标维度, 默认为2。
        
    Returns:
        numpy.ndarray: 降维后的坐标，形状为 (samples, n_components)。
    """
    tsne = TSNE(n_components=n_components, random_state=0, perplexity=30, n_iter=1000)
    low_dim_coords = tsne.fit_transform(features)
    return low_dim_coords

# This funcation calculates the positions of all channels, should be implemented by the participants
def calcLoc(H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num):
    '''
    H: channel data 信道数据 (20000, 100448)
    anch_pos: anchor ID and its coordinates 锚点ID和坐标 (2000, 2)
    bs_pos: coordinate of the base station 基站坐标 ([0, 0, 30])
    tol_samp_num: total number of channel data points 总样本数 (20000)
    anch_samp_num: total number of anchor points 锚点样本数 (2000)
    port_num: number of SRS Ports (number of antennas for the UE) UE天线数 (2)
    ant_num: number of antennas for the base station 基站天线数 (64)
    sc_num: number of subcarriers 子载波数 (408)
    '''
    ######### The following should be implemented by the participants ################

    # Feature Extraction
    # TODO: Apply preprocessing to mitigate Dataset2 impairments, such as AWGN and timing advance.
    test_reduced = extract_features(H)
    real_features = convert_to_real_amplitude_phase(test_reduced)

    # Dimensionality Reduction Mapping
    low_dim_coords = reduce_dimension_tsne(real_features)

    # Transform all the data sets
    train_indices = anch_pos[:, 0].astype(int)
    y_train = anch_pos[:, 1:]

    X_train = low_dim_coords[train_indices]

    mask = np.ones(low_dim_coords.shape[0], dtype=bool)
    mask[train_indices] = False
    X_test = low_dim_coords[mask]

    # Position Prediction
    scaler_X = StandardScaler()
    X_train_std = scaler_X.fit_transform(X_train)
    X_test_std = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_std = scaler_y.fit_transform(y_train)

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_std, y_train_std, test_size=0.1, random_state=42)

    # Definition of NN
    model = Sequential([
        Dense(128, input_shape=(2,)),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        Dense(256),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        Dense(128),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        Dense(2)  # Output layer
    ])

    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Training model
    model.fit(X_train_split, y_train_split, validation_data=(X_val, y_val), epochs=300, batch_size=64, verbose=1, callbacks=[early_stopping])

    # Prediction and inverse normalization
    predict_midnn_std = model.predict(X_test_std)
    predict_midnn_coords = scaler_y.inverse_transform(predict_midnn_std)

    # Generate the final positions
    predicted_coords = np.zeros((20000, 2))
    # indices process
    adjusted_train_indices = train_indices - 1
    predicted_coords[adjusted_train_indices] = y_train

    remaining_indices = np.setdiff1d(np.arange(20000), adjusted_train_indices)
    predicted_coords[remaining_indices] = predict_midnn_coords

    return predicted_coords

# Read in the configuration file
def read_cfg_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        line_fmt = [line.rstrip('\n').split(' ') for line in lines]
    info = line_fmt
    bs_pos = list([float(info[0][0]), float(info[0][1]), float(info[0][2])])
    tol_samp_num = int(info[1][0])
    anch_samp_num = int(info[2][0])
    port_num = int(info[3][0])
    ant_num = int(info[4][0])
    sc_num = int(info[5][0])
    return bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num

# Read in the info related to the anchor points
def read_anch_file(file_path, anch_samp_num):
    anch_pos = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        line_fmt = [line.rstrip('\n').split(' ') for line in lines]
    for line in line_fmt:
        tmp = np.array([int(line[0]), float(line[1]), float(line[2])])
        if np.size(anch_pos) == 0:
            anch_pos = tmp
        else:
            anch_pos = np.vstack((anch_pos, tmp))
    return anch_pos

# The channel file is large, read in channels in smaller slices
def read_slice_of_file(file_path, start, end):
    with open(file_path, 'r') as file:
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines

if __name__ == "__main__":
    print("<<< Welcome to 2024 Wireless Algorithm Contest! >>>\n")
    ## For ease of data managenment, input data for different rounds are stored in different folders. Feel free to define your own
    # PathSet = {0: "./Test", 1: "./Dataset0", 2: "./CompetitionData2", 3: "./CompetitionData3"}
    # PrefixSet = {0: "Round0", 1: "Round1", 2: "Round2", 3: "Round3"}
    PathSet = {0: "./Test", 1: "./Dataset0", 2: "./Dataset1", 3: "./Dataset2"}
    PrefixSet = {1: "Dataset0", 2: "Dataset1", 3: "Dataset2"}

    Ridx = 1  # Flag defining the round of the competition, used for define where to read data。0:Test; 1: 1st round; 2: 2nd round ...
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]
    
    ### Get all files in the folder related to the competition. Data for other rounds should be kept in a different folder  
    files = os.listdir(PathRaw)
    names = []
    for f in sorted(files):
        if f.find('CfgData') != -1 and f.endswith('.txt'):
            names.append(f.split('CfgData')[-1].split('.txt')[0])
    
    
    for na in names:
        FileIdx = int(na)
        print('Processing Round ' + str(Ridx) + ' Case ' + str(na))
        
        # Read in the configureation file: RoundYCfgDataX.txt
        print('Loading configuration data file')
        cfg_path = PathRaw + '/' + Prefix + 'CfgData' + na + '.txt'
        bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num = read_cfg_file(cfg_path)
                
        # Read in info related to the anchor points: RoundYInputPosX.txt
        print('Loading input position file')
        anch_pos_path = PathRaw + '/' + Prefix + 'InputPos' + na + '.txt'
        anch_pos = read_anch_file(anch_pos_path, anch_samp_num)

        # Read in channel data:  RoundYInputDataX.txt
        slice_samp_num = 1000  # number of samples in each slice
        slice_num = int(tol_samp_num / slice_samp_num)  # total number of slices
        csi_path = PathRaw + '/' + Prefix + 'InputData' + na + '.txt'
        H = []
        for slice_idx in range(2): # range(slice_num): # Read in channel data in a loop. In each loop, only one slice of channel is read in
            print('Loading input CSI data of slice ' + str(slice_idx))
            slice_lines = read_slice_of_file(csi_path, slice_idx * slice_samp_num, (slice_idx + 1) * slice_samp_num)
            Htmp = np.loadtxt(slice_lines)
            Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
            Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
            Htmp = np.transpose(Htmp, (0, 3, 2, 1))  # Htmp: (slice_samp_num, ant_num, sc_num, port_num)
            if np.size(H) == 0:
                H = Htmp
            else:
                H = np.concatenate((H, Htmp), axis=0)
        H = H.astype(np.complex64) # trunc to complex64 to reduce storage
        
        csi_file = PathRaw + '/' + Prefix + 'InputData' + na + '.npy'
        np.save(csi_file, H) # After reading the file, you may save txt file into npy, which is faster for python to read 
        # H = np.load(csi_file) # if saved in npy, you can load npy file instead of txt
        
        tStart = time.perf_counter()
        
        
        print('Calculating localization results')
        result = calcLoc(H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num) # This function should be implemented by yourself
        
        # Replace the position information for anchor points with ground true coordinates
        for idx in range(anch_samp_num):
            rowIdx = int(anch_pos[idx][0] - 1)
            result[rowIdx] = np.array([anch_pos[idx][1], anch_pos[idx][2]])

        # Output, be careful with the precision
        print('Writing output position file')
        with open(PathRaw + '/' + Prefix + 'OutputPos' + na + '.txt', 'w') as f:
            np.savetxt(f, result, fmt='%.4f %.4f')

        # This help to evaluate the running time, can be removed!
        tEnd = time.perf_counter()
        print("Total time consuming = {}s\n\n".format(round(tEnd - tStart, 3)))