from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from sklearn.mixture import GaussianMixture
from scipy.special import softmax
import joblib
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

########################################################################
################################# GUI ##################################
########################################################################

def launch_main_window():
	def add_widgets_train(frame):
		def browse():
			foldername = filedialog.askdirectory()
			library_path.set(foldername)
			print('Library:', foldername)

		def train():
			library = library_path.get()
			if check_library_trainable(library):
				messagebox.showinfo('Error', 'The path supplied is not valid. Please provide a valid path.')
				return
			library = Path(library)
			buildUBM(library, 16)
			adaptUBM(library)
			print('Training Complete!')
			print(library, 'is ready for use')
			messagebox.showinfo('Done!', "The Training is Complete!")

		library_path = StringVar()
		
		Label(frame, height='2').pack()
		Label(frame, text="Select a library. Enter path or browse to select folder.").pack()

		path_frame = Frame(frame)
		path_frame.pack(pady='20')

		Label(path_frame, text='Path to Library:', width='17', height='2').pack(side=LEFT)
		pathentry = Entry(path_frame, textvariable=library_path, width='55')
		pathentry.pack(side=LEFT)
		Button(path_frame, text='Browse', command=browse, width=5).pack(side=LEFT)

		Button(frame, text='Train on Library', width='20', height='2', command=train).pack()

	def add_widgets_stest(frame):
		def browseLib():
			foldername = filedialog.askdirectory()
			library_path.set(foldername)
			print('Library:', foldername)

		def browseAudio():
			filename = filedialog.askopenfilename()
			file_path.set(filename)
			print('Selected Audio File:', filename)

		def stest():
			audio = file_path.get()
			if check_valid_audio(audio):
				messagebox.showinfo('Error', 'The audio supplied is not valid. Please provide a valid audio file.')
				return
			library = library_path.get()
			if check_library_trained(library):
				messagebox.showinfo('Error', 'The library supplied is not valid. Please provide a valid library.')
				return
			audio = Path(audio)
			library = Path(library)
			result_label.config(text='Searching...')
			result = identify_speaker_from_library(audio, library)
			d = analysis(audio, library)
			result = d.Name[d.Score.idxmax()]
			print(d)
			result_label.config(text=audio.stem+' belongs to '+result+' of '+library.stem)

		file_path = StringVar()
		library_path = StringVar()

		Label(frame, height='2').pack()
		Label(frame, text="Select an Audio File. Enter path or browse to select file.").pack()

		file_path_frame = Frame(frame)
		file_path_frame.pack(pady='20')

		Label(file_path_frame, text='Path to Audio File:', width=22, height=2).pack(side=LEFT)
		pathentry = Entry(file_path_frame, textvariable=file_path, width=50)
		pathentry.pack(side=LEFT)
		Button(file_path_frame, text='Browse', command=browseAudio, width=5).pack(side=LEFT)
		
		Label(frame, height='2').pack()
		Label(frame, text="Select a trained library. Enter path or browse to select folder.").pack()

		path_frame = Frame(frame)
		path_frame.pack(pady='20')

		Label(path_frame, text='Path to Library:', width=22, height=2).pack(side=LEFT)
		pathentry = Entry(path_frame, textvariable=library_path, width=50)
		pathentry.pack(side=LEFT)
		Button(path_frame, text='Browse', command=browseLib, width=5).pack(side=LEFT)

		Button(frame, text='Identify', command=stest).pack(pady=10)

		result_label = Label(frame, text='', width='300', height='2')
		result_label.pack()

	def add_widgets_btest(frame):
		def browseLib():
			foldername = filedialog.askdirectory()
			library_path.set(foldername)
			
		def browse_testdir():
			foldername = filedialog.askdirectory()
			testdir_path.set(foldername)

		def btest():
			testdir = testdir_path.get()
			if check_valid_testdir(testdir):
				messagebox.showinfo('Error', 'The test folder supplied is not valid. Please provide a valid test folder.')
				return
			library = library_path.get()
			if check_library_trained(library):
				messagebox.showinfo('Error', 'The library supplied is not valid. Please provide a valid library.')
				return
			testdir = Path(testdir)
			library = Path(library)
			name, result, score, confidence = list(), list(), list(), list()
			for p in testdir.iterdir():
				if check_valid_audio(p):
					print('skipping', p)
					continue
				name.append(p.name)
				d = analysis(p, library)
				idx = d.Score.idxmax()
				result.append(d.Name[idx])
				score.append(d.Score[idx])
				confidence.append(d.Confidence[idx])
			d = pd.DataFrame({'File_Name':name, 'Prediction':result, \
					'Score':score, 'Confidence':confidence})
			d.sort_values('File_Name', inplace=True)
			d.reset_index(drop=True, inplace=True)
			d.to_csv(testdir/(testdir.name+'_predictions.csv'), index=False)
			messagebox.showinfo('Done!', 'Finished Bulk Test!')

		testdir_path = StringVar()
		library_path = StringVar()

		Label(frame, height='2').pack()
		Label(frame, text="Select a Test Folder. Enter path or browse to select folder.").pack()

		testdir_path_frame = Frame(frame)
		testdir_path_frame.pack(pady='20')

		Label(testdir_path_frame, text='Path to Test Folder:', width=22, height=2).pack(side=LEFT)
		pathentry = Entry(testdir_path_frame, textvariable=testdir_path, width=50)
		pathentry.pack(side=LEFT)
		Button(testdir_path_frame, text='Browse', command=browse_testdir, width=5).pack(side=LEFT)
		
		Label(frame, height='2').pack()
		Label(frame, text="Select a trained library. Enter path or browse to select folder.").pack()

		path_frame = Frame(frame)
		path_frame.pack(pady='20')

		Label(path_frame, text='Path to Library:', width=22, height=2).pack(side=LEFT)
		pathentry = Entry(path_frame, textvariable=library_path, width=50)
		pathentry.pack(side=LEFT)
		Button(path_frame, text='Browse', command=browseLib, width=5).pack(side=LEFT)

		Button(frame, text='Identify', command=btest).pack(pady=10)

		result_label = Label(frame, text='', width='300', height='2')
		result_label.pack()

	def add_widgets_perf_check(frame):
		def browseLib():
			foldername = filedialog.askdirectory()
			library_path.set(foldername)
			
		def browse_testdir():
			foldername = filedialog.askdirectory()
			testdir_path.set(foldername)

		def perfcheck():
			library = library_path.get()
			if check_library_trained(library):
				messagebox.showinfo('Error', 'The library supplied is not valid. Please provide a valid library.')
				return
			testdir = testdir_path.get()
			testdir = Path(testdir)
			library = Path(library)

			conmat = perf_test(library, testdir)
			print(conmat)
			messagebox.showinfo('Done!', 'Finished Performance Test!')

		testdir_path = StringVar()
		library_path = StringVar()
		
		Label(frame, height='2').pack()
		Label(frame, text="Select a trained library. Enter path or browse to select folder.").pack()

		path_frame = Frame(frame)
		path_frame.pack(pady='20')

		Label(path_frame, text='Path to Library:', width=22, height=2).pack(side=LEFT)
		pathentry = Entry(path_frame, textvariable=library_path, width=50)
		pathentry.pack(side=LEFT)
		Button(path_frame, text='Browse', command=browseLib, width=5).pack(side=LEFT)

		Label(frame, height='2').pack()
		Label(frame, text="Select a Test Folder. Enter path or browse to select folder.").pack()

		testdir_path_frame = Frame(frame)
		testdir_path_frame.pack(pady='20')

		Label(testdir_path_frame, text='Path to Test Folder:', width=22, height=2).pack(side=LEFT)
		pathentry = Entry(testdir_path_frame, textvariable=testdir_path, width=50)
		pathentry.pack(side=LEFT)
		Button(testdir_path_frame, text='Browse', command=browse_testdir, width=5).pack(side=LEFT)

		Button(frame, text='Check Performance', command=perfcheck).pack(pady=10)

		result_label = Label(frame, text='', width='300', height='2')
		result_label.pack()


	main_window = Tk()
	main_window.title('Speaker Recognition')
	main_window.geometry('700x350')

	nb = ttk.Notebook(main_window, height=300, width=650, padding=10)

	train_frame = ttk.Frame(nb)
	add_widgets_train(train_frame)
	nb.add(train_frame, text='      Training      ')

	stest_frame = ttk.Frame(nb)
	add_widgets_stest(stest_frame)
	nb.add(stest_frame, text='      Single Test      ')

	btest_frame = ttk.Frame(nb)
	add_widgets_btest(btest_frame)
	nb.add(btest_frame, text='      Bulk Test      ')

	perfcheck_frame = ttk.Frame(nb)
	add_widgets_perf_check(perfcheck_frame)
	nb.add(perfcheck_frame, text='      Performance Check      ')

	nb.pack(expand=True, fill='both')
	main_window.mainloop()


########################################################################
############################# Backend ##################################
########################################################################
	
supported_audio_formats = ['wav', 'mp3']

def check_valid_testdir(testdir):
	print('Testdir Validity Check on:', testdir)
	if str(testdir) == '':
		print('Test directory path provided must not be empty.')
		return -1
	testdir = Path(testdir)
	if not testdir.is_dir():
		print(testdir, 'must be a directory.')
		return -2
	for p in testdir.iterdir():
		if p.is_dir() or p.stem.startswith(testdir.stem):
			print('skipping', p)
			continue
		if check_valid_audio(p):
			print(p, 'is not valid audio.')
			return -3
	return 0

def check_valid_audio(audio, verbose=1):
	if str(audio) == '':
		print('Audio Path is not valid.') if verbose else False
		return -1
	audio = Path(audio)
	if not audio.is_file():
		print(audio, 'is not a file.') if verbose else False
		return -2
	if not np.array([audio.suffix.endswith(ext) for ext in supported_audio_formats]).any():
		print(audio, 'doesn\'t end with a supported suffix.') if verbose else False
		print('supported audio formats are:', supported_audio_formats) if verbose else False
		return -3
	return 0

def check_library_trained(library):
	if str(library) == '':
		print('Library path provided is empty.')
		return -1
	library = Path(library)
	if not library.is_dir():
		print('Library Path provided is not a directory.')
		return -1
	if not (library/(library.stem+'_ubm')).is_file():
		print((library/(library.stem+'_ubm')), 'is not a file.')
		return -2
	for person in library.iterdir():
		if not person.is_dir():
			continue
		if not (person/(person.stem+'_gmm')).is_file():
			print((person/(person.stem+'_gmm')), 'is not a file')
			return -3
	return 0

def check_library_trainable(library):
	if str(library) == '':
		print('Library path provided is empty.')
		return -1
	library = Path(library)
	if not library.is_dir():
		print('Library Path provided is not a directory.')
		return -1
	for p in library.iterdir():
		if p.is_file() and not p.stem.startswith(library.stem):
			print(p, 'Should not exist')
			return -2
	for p in library.iterdir():
		if p.stem.startswith(library.stem):
			continue
		for p1 in p.iterdir():
			if p1.is_dir():
				print(p, 'Should not exist')
				return -3
	return 0

def perf_test(train, test):
	'''A function that generates a confusion matrix for a given pair of
	train and test library.

	Parameters:
	train: Path to a trained library
	test: Path to the testing library

	Returns:
	ConMat: Confusion Matrix in the form of a pandas DataFrame.

	Warning: Performs no sanity check on the parameters provided.
	'''
	ubm, people, names = loadlibrary(train, sanity = True)
	conmat = initconmat(names)
	for name in names:
		for p in (test/name).iterdir():
			if check_valid_audio(p, verbose=0):
				print('skipping', p)
				continue
			conmat[name][identify(p, ubm, people, names)] += 1
	return conmat

def identify(audio, ubm, people, names):
	return names[analyse(_extract_features_file(audio), people, 
		ubm).argmax()]

def loadlibrary(library, sanity=False):
	'''A function that returns the ubm, the list of adapted models, and
	their corresponding names from a library provided.

	Parameters:
	library: Path to a trained library.
	sanity: default False. Whether or not to perform sanity check on
	argument provided.
	
	Returns:
	ubm: The Universal Background Model
	people: A list of the adapted models.
	names: A list of the names of people in the library. Index here
	matches the index in people.
	'''
	ubm = joblib.load(library/(library.stem+'_ubm'))
	people, names = list(), list()
	for p in library.iterdir():
		if not p.is_dir():
			continue
		names.append(p.stem)
		people.append(joblib.load(p/(p.stem+'_gmm')))
	return ubm, people, names

def initconmat(names):
	'''Initializes a Confusion Matrix in the form of a pandas DataFrame

	Paramters:
	names: A list of names of the people involved in the library.

	Returns:
	conmat: An initialized confusion matrix.
	'''
	n = len(names)
	return pd.DataFrame(np.zeros((n,n), int), index=names, 
		columns=names)

def _people_of_library(library):
	'''Return a list of names of the people in the trained library. This
	doesn't do any sanity check on the library path provided.

	Parameters:
	library: Path to library

	Returns:
	people: a List of names of people in the library.
	'''
	library = Path(library)
	people = List()
	for p in library.iterdir():
		if p.is_dir():
			people.append(p.stem)
	return people

def _extract_features_file(file):
	'''Extract the 39 dimensional features from a single file.
	
	Parameters:
	file: path to audio file. Look at `supported_audio_formats` for accepted 
		audio formats.

	Returns:
	features: ndarray (n, 39) 39 dimensional feature vectors extracted 
		from the single file 
	'''
	y, sr = librosa.load(file, sr=None)
	hop_length = (sr*20)//1000
	f = librosa.feature.mfcc(y, sr=sr, hop_length=hop_length, n_mfcc=13)
	return np.vstack((f, 
					librosa.feature.delta(f, order=1, axis=0), 
					librosa.feature.delta(f, order=2, axis=0))
					).T

def _extract_features_folder(folder):
	'''Extract all the features from all the audio files present in a 
	single folder.

	Parameters:
	folder: path to folder containing audio files

	Returns:
	features: ndarray (n,39) - all the features from all the files
		present in the folder.
	'''
	return np.concatenate([_extract_features_file(file) 
		for extension in supported_audio_formats 
		for file in folder.glob('*.'+extension)], axis=0)

def extract_features_folder(folder):
	'''Extract and store the features that can be extracted from a
	single folder. The features will be stored in npz compressed 
	archive in the same folder. The features can be accessed as 'f'.
	The file will be titled with the same name as the stem of 
	`folder`.

	Parameters:
	folder: path to folder containing audio files

	Returns:
	None
	'''
	np.savez_compressed(folder/folder.stem, 
		f=_extract_features_folder(folder))

def _ensure_all_features(library):
	'''Extract and store all the features from the people in the library

	Parameters:
	library: path to library

	Returns:
	None

	Note: This will not compile all the features. It will just ensure 
		all the people in the library have their features extracted.
	'''
	library = Path(library)
	for person in library.iterdir():
		if not person.is_dir():
			continue
		print('extracting features of', person.stem)
		extract_features_folder(person)

def compile_all_features(library):
	'''Extract and compile all the features of all people in a library.
	After extracting features of each person, it will then compile all
	of them, save it in the library as npz compressed file with the same
	name.

	Parameters:
	library: path to library

	Returns:
	None
	'''
	library = Path(library)
	print('extracting all features from', library)
	_ensure_all_features(library)
	np.savez_compressed(library/library.stem, 
		f=np.concatenate([np.load(person/(person.stem+'.npz'))['f'] 
			for person in library.iterdir() if person.is_dir()], 
			axis=0)
		)

def _get_features_library(library):
	'''Simply return all the features of the library. If features of the
	library are not compiled, this will fail.

	Parameters:
	library: path to library

	Returns:
	features: compiled features of every person in library
	'''
	library = Path(library)
	return np.load(library/(library.stem+'.npz'))['f']

def get_features_library(library):
	'''Extract all the features from a library. Meant to be used for 
	building UBM.

	Parameters:
	library: path to library

	Returns:
	features: compiled features from all people in library.
	'''
	print('compiling features')
	compile_all_features(library)
	print('loading compiled features')
	return _get_features_library(library)

def ubm(features, nmixtures, cov_type='diag'):
	'''Trains a GMM-UBM with certain number of mixtures on a given set 
	of features.

	Parameters:
	features: ndarray of features
	nmixtures: int, number of mixtures in GMM

	Returns:
	gmm: a trained GMM-UBM

	Note: The model has the attribute warm_start set to True. Subsequent
	calls to fit() will begin from these params.
	'''
	gmm = GaussianMixture(n_components=int(nmixtures), 
		covariance_type=cov_type, max_iter=150, warm_start=True)
	gmm.fit(features)
	return gmm

def buildUBM(library, nmixtures):
	'''Builld a UBM for a library, while saving all the files needed for
	adapting the ubm for specific people. Saves the ubm in the library 
	with the same name with '_ubm' suffix

	Paramters:
	library: path to library
	nmixtires: int, the number of mixtures in GMM
	'''
	library = Path(library)
	print('getting features from library')
	features = get_features_library(library)
	print('training ubm')
	gmm_ubm = ubm(features, nmixtures)
	print('saving ubm')
	joblib.dump(gmm_ubm, library/(library.stem+'_ubm'))

def _adaptUBM(ubm, X):
	'''Adapts a single UBM to given set of features. This functions is 
	not pure and will modify the UBM provided. Make sure to pass a fresh
	ubm everytime.

	Parameters:
	ubm: A pretrained sklearn GaussianMixture object.
	X: The set of features to be adapted upon

	Returns:
	gmm: An adapted GaussianMixture Object.
	'''
	# return ubm.fit(X)

	probablistic_alignment = ubm.predict_proba(X)
	'''predict_proba gives the posterior probability for each component.
	This is nothing but Responsibility, as it can interpreted as 
	posterior probability. Posterior probability is termed as the 
	probabilistic alignment in the paper.
	'''

	weights = np.sum(probablistic_alignment, axis=0)
	means = np.matmul(probablistic_alignment.T, X) \
		/ weights.reshape((-1,1))
	variances = np.matmul(probablistic_alignment.T, X**2) \
		/ weights.reshape((-1,1)) 
	'''These 3 parameters: the weight, mean and variance, together are 
	termed as sufficient statistics in the paper.
	'''

	relevance_factor = 14
	alpha = weights / (weights + relevance_factor)
	new_weights = (alpha*weights)/X.shape[0] + (1-alpha)*ubm.weights_
	new_weights /= np.sum(new_weights) # Normalize it
	new_means = alpha.reshape((-1,1)) * means + (1-alpha).reshape((-1,1)) * ubm.means_
	new_covariances = alpha.reshape((-1,1)) * variances \
		+ (1-alpha).reshape((-1,1))*(ubm.covariances_**2 + ubm.means_**2) \
		- new_means**2

	ubm.weights_ = new_weights.copy()
	ubm.means_ = new_means.copy()
	ubm.covariances_ = new_covariances.copy()
	return ubm

def adaptUBM(library):
	'''For a library with an existing UBM, and other neccessary file, 
	this function adapts the UBM to every person present in the library.
	The adapted model is saved in each person's folder.

	Parameters:
	library: path to library

	Returns:
	None
	'''
	library = Path(library)
	for person in library.iterdir():
		if not person.is_dir():
			continue
		print('adapting ubm to', person.stem)
		ubm = joblib.load(library/(library.stem+'_ubm'))
		person_features = np.load(person/(person.stem+'.npz'))['f']
		model = _adaptUBM(ubm, person_features)
		joblib.dump(model, person/(person.stem+'_gmm'))

def analyse(subject, people, ubm):
	'''Analyse the given subject with regard to the given list of people
	and the respective ubm. It calculates the final posterior 
	probabilities and the per person scores according to the reynolds 
	paper.

	Parameters:
	subject: The features extracted from the subject
	people: An iterable list of adapted models
	ubm: The UBM from which the adapted models were made.

	Returns:
	confidence: Posterior probabilities of classification.
	per_person_scores: The scores as defined in the reynolds paper.
	'''
	per_person_scores = []
	for person in people:
		per_person_scores.append(np.sum(person.score_samples(subject)))
	per_person_scores = np.array(per_person_scores)
	per_person_scores -= np.sum(ubm.score_samples(subject))
	return per_person_scores

def analysis(audio_file, library):
	'''Perform analysis on the audio file provided.

	Parameters:
	audio_file: path to audio file
	library: path to library with gmm of each person.

	Returns:
	names: The names of the people in the library.
	confidence: Posterior probabilities of classification.
	per_person_scores: The scores as defined in the reynolds paper.
	'''
	subject_features = _extract_features_file(audio_file)
	people_names = []
	people_models = []
	library = Path(library)
	ubm = joblib.load(library/(library.stem+'_ubm'))
	for person in library.iterdir():
		if not person.is_dir():
			continue
		if (person/(person.stem+'_gmm')).is_file():
			people_names.append(person.stem)
			people_models.append(joblib.load(person/(person.stem+'_gmm')))
	score = analyse(subject_features, people_models, ubm)
	confidence = softmax(score)
	return pd.DataFrame({'Name':people_names, 'Score':score, 'Confidence':confidence})	

def identify_speaker_from_library(audio_file, library):
	'''Identify the speaker from a given wave file and a library.

	Parameters:
	audio_file: path to audio file
	library: path to library with gmm of each person.

	Returns:
	Name: Name of the person in the library to whom this wav file 
	belongs to.
	'''
	d = analysis(audio_file, library)
	return d.Name[d.Score.idxmax()]

def backend(path):
	buildUBM('Data', 16)
	adaptUBM('Data')
	print(identify_speaker_from_library('test/p2/sx178.wav', 'Data'))

def backendtester():
	p = Path('Data')
	backend(p)

def frontendtester():
	launch_main_window()

if __name__ == '__main__':
	# backendtester()
	frontendtester()