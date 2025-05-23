This documentation file was generated on 2022-06-16 by Jose Yauri



GENERAL INFORMATION
-------------------

1. Title of Dataset: Dataset of physiological data to predict mental workload.


2. Authorship: 
	Name: Jose Yauri
	Institution: Centre de Visió per Computador (CVC), Universitat Autònoma de Barcelona
	Email: jyauri@cvc.uab.cat
	ORCID: 0000-0001-6287-7797

	Name: Pau Folch
	Institution: Aslogic, Parc de Recerca Universitat Autònoma de Barcelona
	Email: pau.folch@uab.cat
	ORCID: -

	Name: Daniel Álvarez
	Institution: Aslogic, Parc de Recerca Universitat Autònoma de Barcelona
	Email: dalvarez@aslogic.es
	ORCID: -

	Name: Débora Gil Resina
	Institution: Centre de Visió per Computador (CVC), Universitat Autònoma de Barcelona
	Email: debora@cvc.uab.cat
	ORCID: 0000-0002-2770-4767

	Name: Aura Hernández-Sabaté
	Institution: Centre de Visió per Computador (CVC), Universitat Autònoma de Barcelona
	Email: aura@cvc.uab.cat
	ORCID: 0000-0003-1563-9934



DESCRIPTION
-----------

1. Dataset language: English


2. Abstract:
A high mental workload reduces the human performance and affects his/her ability to achieve a task. Despite the recent advances in neuroscience, yet there is a lack of knowledge about the interrelation between the mental processes in the brain and the produced mental workload at a giving time. The use of neuro-physiological data to assess abnormal mental states in the last decade has led to new manners to explore the brain and its association with mental workload. We present an open dataset for mental workload investigation. The dataset contains neuro-physiological recordings collected using an electroencephalogram (EEG) and an electrocardiogram (ECG). Participants were submitted to different tasks under different conditions to induce different levels of workload. 
In particular, three subsets were collected. First, playing the N-back test game to enforce the short term memory. Second, playing the Heat-the-Chair game (a serious game of own design) which enforce the performing of simultaneous tasks. Third, flying in an immersive simulated Airbus320 cockpit environment. The pilot must solve diverse critical situations, such as an engine failure, a sudden wind shear, or an urgent call of the air traffic controller (ATC). 
The design of the datasets has been validated by correlating the performance of subjects to their self-perceived difficulty.
To make the dataset useful for testing the experiments, the ground-truth of mental workload of each task, both the objective and the subjective self-perceived is provided.


3. Keywords: Cognitive states, Mental workload prediction, Neuro-physiological data, EEG, ECG, Neural Networks.

4. Date of data collection (single date or date range): From 2020-10-01 to 2021-07-30

5. Date of dataset publication: 2022-06-06

6. Funding sources: 

	Funding agency: H2020 Clean Sky 2 project: "E-PILOT: Evolution of cockPIt operations Levering on cOgnitive compuTing Services"
	Project number: Grant Agreement No. 831993

7. Geographic location/s of data collection:

	Barcelona, Catalonia, Spain



ACCESS INFORMATION
------------------

1. Creative Commons License of the dataset: CC BY-SA

2. Dataset DOI:
[This information will be filled in by the DDD team after validating data.]

3. Related publication:

Hernández-Sabaté, A., Yauri, J., Folch, P., Piera, M. À., & Gil, D. (2022). Recognition of the Mental Workloads of Pilots in the Cockpit Using EEG Signals. Applied Sciences 2022, Vol. 12, Page 2298, 12(5), 2298. https://doi.org/10.3390/APP12052298

Yauri, J., Hernández-Sabaté, A., Folch, P., & Gil, D. (2021). Mental Workload Detection Based on EEG Analysis. Frontiers in Artificial Intelligence and Applications, 339, 268–277. https://doi.org/10.3233/FAIA210144



VERSIONING AND PROVENANCE
-------------------------

1. Last modification date: 2022-06-22

2. Was data derived from another source?: No

3. Additional related data collected that was not included in the current data package: None


METHODOLOGICAL INFORMATION
--------------------------

When a subject performs tasks along the day, (s)he experiments different levels of mental workload depending on the level of cognition required to achieve the tasks. Mental workload has direct effect on subject's performance, and the chances to commit mistakes increase when the workload is too high. So, it has become a hot topic of study to discover if exists correlation between the neuro-physiological responses of a subject when facing challenges of different mental workload degree. In order to study mental workload a set of experiments have been designed, in which we record neuro-physiological signals using an electroencephalogram (EEG) and an electrocardiogram (ECG) while the volunteer performs exercises that induce mental workload from easy to hard level. Besides, we also registered the self-perceived workload and the scores obtained in the task. Moreover, each experiment is labeled according to the degree of workload induced.

1. Description of methods used for collection-generation of data: 

To collect data, we proposed four experiments, grouped in two types and described as follows:


a) The N-back-test: In this first experiment, the volunteer faces three variants of the N-Back-test to induce low, medium, and high mental workload. This game is intensive in short term memory usage, so affects the subject performance for success in solving the task.

Task 1 [low workload]: the position 1-back. A highlighted square appears every few seconds in one of eight different positions over a regular grid on the screen. The subject must press a key to confirm if the position of the current square matches the square's position shown in the trial before.
    
Task 2 [medium workload]: the arithmetic 1-back. A number between 0 and 9 appears every few seconds on the screen, together with an audio message indicating an arithmetic operation. The subject must write the result of the operation between the current number and the number shown before. The four arithmetic operations randomly appear.
    
Task 3 [high workload]: the dual arithmetic 2-back. This experiment combines the two previous ones, but taking in account both the square and the number that appeared two trial before. 

16 volunteers participated in this experiment, collecting 48 recording sessions. Besides the neuro-physiological data, we provide the game scores, and the TLX-questionnaire with the self-subject perceived workload degree.


b) The Heat-the-Chair game: In this second experiment, we implemented a game to emphasize the making of simultaneous tasks, which require concentration and alertness. We tried to mimic the activities performed by pilots while flighting a plane. The game is computer-based, so the subject must use the keyboard and mouse for gaming. 

In order the differentiate the degree of workload produced by the game, it has two modes:

Task 1 [low workload]: game without interruptions. Here the gamer must maintain the flight stability and perform some tasks specific to the game

Task 2 [medium workload]: game with interruptions. In addition to the tasks presented in Task 1, the gamer must attend the interruptions that replicate the interactions with the air traffic controller (ATC).

If the player loses concentration, collapses into instability or does not solve asked tasks of ATC, the game penalizes the score and turns the game a much harder to finalize the task. 

17 volunteers participated in this experiment, collecting 34 recording sessions. We provide the neuro-physiological data, the game scores, and the answered TLX-questionnaires.


c) Data from flight simulation: In these experiments we carried out a set of flight missions in an Airbus320 cockpit with the participation of two professional pilots. 

The flight simulation consists of five flight missions and the goal was to evaluate the pilot workload changes while he resolves unexpected flight situations. Two pilots participated in all flights, however they interchanged roles: either one acts as the pilot or as an observer. 

The flight scenarios are presented as follows:

 - Flight 1 [easy difficulty]: the pilot executes a simple standard flight to be used as the reference parameters. 

 - Flight 2 [medium difficulty]: in an unexpected moment of the flight, the ATC reports a much traffic and commanded the pilot to change the airplane direction as quick as possible.
   
 - Flight 3 [hard difficulty]: at the final stage of the flight, the airplane is strongly destabilized by a wind shear, so pilot must maneuver for landing.
    
 - Flight 4 [medium difficulty]: at the middle of the flight, it happens a malfunction as a result of a human error on controls, which cause the failure of an engine and collapsing the crew workload.

- Flight 5 [medium difficulty]: it is similar to Flight 2. but with a little variation.

Two pilots, aging 32 and 51 years, with 1700 and 4000 flight hours, respectively, participated in these experiments. We provide the neuro-physiological data and the difficulty degree of each flight.

Further information about the methods can be found in our publications presented above.


2. Methods for processing the data: 

The EEG used in all experiments was a 14-electrodes Emotiv Epoc X. For ECG, we used the Suunto Ambit3 Peak with a hearth rate belt for the N-back-test and the flight simulation, whereas the Shimmer3 was used for the Heat-The-Chair game. These devices provide their own software utility to record and save the recording data as continuous time series data. Those sofwares make post processing for us. Besides the 14-electrode data, the Emotiv also provides the power band of four main brain frequencies (theta, alpha, beta_low, beta_high, and gamma). The Suunto ECG provides the Hearth Rate Variability (HR), and the Interbeat Interval (IBI), whereas the Shimmer3 computed for us the HR, IBI, and the Breathing Rate (BR).  All recordings are continuous in time.

To generate the data contained in these datasets, we only took such continuous data and segmented it according to the experiments, adding additional metadata and information for easy data retrieval. For such data generation, we built a set of Python scripts. No filtering nor noise removal method was applied, so the dataset contains the raw data of the experiments. The outputs of Python scripts are dataframes. A dataframe is tabular data structure, accessible from any language programming. 


3. Instrument- or software- specific information needed to interpret the data:

The datasets can be opened, explored and processed by any language programing, with the Dataframe utility. For instance, in Python 3.8+, using Pandas library; or in Matlab 2019b+, reading the file parquet directly.


4. Instruments, calibration and standards information:

No especial instrument of calibrator was used.

5. Environmental or experimental conditions:

The game experiments were performed in a normal room, with a computer running the game and the subject playing with the game while wearing the EEG and ECG. Both the EEG and ECG transmit the collected data via Bluetooth to another computer for recording. 

The flight simulation experiment was carried out in an Airbus320 cockpit simulator, especially conditioned in the El Prat Airport, Barcelona, Spain.


6. Quality-assurance procedures performed on the data:

To guarantee the quality of collected data, before each experiment, the connectivity and conductivity of sensors were verified using their own software utility.



FILE OVERVIEW
--------------

1. Explain the file naming convention, if applicable:

As exposed above, EEG and ECG data are stored in dataframes, so they have the extension ‘.parquet’. An EEG file has a prefix "eeg", whilst an ECG, has a prefix "ecg".

Experiments are stored into separated folders that have the prefix "data", followed by the name of the experiment. Into each folder, data are organized into subfolders following a self-explainable name.


2. File List:


.\workload_dataset
	.\data_flight_simulation

		.\ecg
			ecg_hr.parquet
			ecg_ibi.parquet
		.\eeg
			eeg.parquet

		.\perceived_difficulty
			flight_1.json
			flight_2_4.json
			flight_3_5.json

        .\data_heath_the_chair
		.\ecg
			ecg.parquet
		.\eeg
			eeg.parquet
		.\game_performance
			subject_01_with.csv
			subject_01_without.csv
 			subject_02_with.csv
 			subject_02_without.csv
			subject_03_with.csv
 			subject_03_without.csv
 			subject_06_with.csv
 			subject_06_without.csv
 			subject_08_with.csv
 			subject_08_without.csv
 			subject_12_with.csv
 			subject_12_without.csv
 			subject_16_with.csv
 			subject_16_without.csv
 			subject_17_with.csv
 			subject_17_without.csv
 			subject_18_with.csv
 			subject_18_without.csv
 			subject_19_with.csv
 			subject_19_without.csv
 			subject_20_with.csv
 			subject_20_without.csv
 			subject_21_with.csv
 			subject_21_without.csv
 			subject_22_with.csv
 			subject_22_without.csv
 			subject_23_with.csv
 			subject_23_without.csv
 			subject_24_with.csv
 			subject_24_without.csv
 			subject_25_with.csv
 			subject_25_without.csv
 			subject_26_with.csv
 			subject_26_without.csv				
		.\subjective_performance
			tlx_answers.parquet

	.\data_n_back_test
		.\ecg
			ecg_br.parquet
			ecg_hr.parquet
			ecg_ibi.parquet
		.\eeg
			eeg.parquet
		.\game_performance
			game_scores.parquet
		.\subjective_performance
			tlx_answers.parquet

3. Relationship between files:

Each experiment is independent from the others and collected in different times. However, it is worth to mention that some subjects have participated in both the N-back test and the Heat-the-Chair game, they can be identified the subject id.


4. File format:

File dataframe parquet (.PARQUET), comma separated (.CSV), and json data files (.JSON).


MORE INFORMATION 
----------------
None
