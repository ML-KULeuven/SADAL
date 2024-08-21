import numpy as np
from skopt import gp_minimize
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ssif import SSIF
import seaborn as sns
from matplotlib.patches import Patch

class SADAL:
    def __init__(self, Xtrain, ytrain, contamination,  gamma=0.5, c_fp=1, c_fn=1, c_r=None, seed=9):
        """
        SADAL class
        The SADAL algorithm combines Active Learning and Learning to Reject to reduce the mispredictions of a semi-supervised anomaly detector.
        
        Parameters
        ----------
        Xtrain : numpy array of shape (n_training_samples+n_validation_samples, n_features)
            The input samples.
        ytrain : numpy array of shape (n_training_samples+n_validation_samples)
            The input true labels.
        contamination : float, optional
            The number of SSiTree in the ensemble.
        gamma : float, optional 
           The weights for the past allocation rounds in the reward function.
        c_fp : float, optional
            The cost of a false positive.
            If not specified, it is set to 1 by default.
        c_fn : float, optional
            The cost of a false negative.
            If not specified, it is set to 1 by default.
        c_r : float, optional
            The cost of a rejection.
            If not specified, it is set to min(c_fp*(1-contamination), c_fn*contamination) by default.
        seed : int, optional
            random_state is the seed used by the random number generator;
            If None, it is set to 9 by default
        """
        self.Xtrain, self.Xval, self.ytrain, self.yval = train_test_split(Xtrain, ytrain,
                                                    test_size=0.5,
                                                    stratify=ytrain,
                                                    shuffle=True,
                                                    random_state=seed)
        self.contamination = contamination
        self.gamma = gamma
        self.c_fp = c_fp
        self.c_fn = c_fn
        self.c_r = min(c_fp*(1-contamination), c_fn*contamination) if c_r == None else c_r
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        
    def run_allocation_loop(self, nrounds, tot_budget, Xtest, ytest, plots = False):
        """
        This function performs an allocation loop to allocate a budget B. At each round, the labels for training or validation instances are queried to the oracle depending on the rewards for the past rounds.
        
        Parameters
        ----------
        nrounds : int 
           The number of allocation rounds to perform;
        tot_budget : int
            The budget to allocate;
        Xtest : numpy array of shape (n_samples, n_features)
            The test samples;
        ytest : numpy array of shape (n_samples)
            The test true labels;
        plots : bool, optional
            Whether to plot the intermediate results or not;
            It is set to False by default;
        """
        y_train_semitargets = np.zeros(len(self.ytrain), dtype=int) # training labels, initially unlabeled set
        y_val_semitargets = np.zeros(len(self.yval), dtype=int) # validation labels, initially unlabeled set

        pool_budget = int(tot_budget/nrounds)
        
        self.nroundLR = self.nroundAL = 0
        self.ALrewards = -1*np.ones((nrounds,))
        self.LRrewards = -1*np.ones((nrounds,))
        self.LRrewards[0] = 1 
        self.ALrewards[0] = 0.99
        
        self.initialize_loop(np.vstack([self.Xtrain, self.Xval]))
        
        test_costs = np.zeros((nrounds,))
        fprs = np.zeros((nrounds,))
        fnrs = np.zeros((nrounds,))
        rejection_rate = np.zeros((nrounds,))
        rejected = []
        for idx in range(nrounds):
            detector, thresholds = self.run_allocation_round(self.Xtrain, y_train_semitargets, self.Xval, y_val_semitargets, pool_budget, rejected)
            test_costs[idx], fprs[idx], fnrs[idx], rejection_rate[idx], probs = self.evaluate_model(Xtest, ytest, detector, thresholds)
            rejected = np.where(np.logical_and(self.train_probs >= thresholds[0], self.train_probs <= thresholds[1]), 1, 0)[0]

            if plots:
                plot_SADAL(Xtest, ytest, detector, probs, thresholds, idx+1)
        return test_costs, fnrs, fprs, rejection_rate

    def run_allocation_round(self, Xtrain, ytrain, Xval, yval, budget, rejected = []):
        """
        This function performs an allocation round to allocate the budget. 
        First, it checks which option has the highest reward and then queries the labels to the oracle.
        After acquirying new labels, the function fits the detector and optimizes the rejection thresholds. Finally, the rewards are updated.
        
        Parameters
        ----------
        Xtrain : numpy array of shape (n_training_samples, n_features)
            The training samples;
        ytrain : numpy array of shape (n_training_samples)
            The training labels (0:unlabeled, 1:anomaly, -1:normal);
        Xval : numpy array of shape (n_validation_samples, n_features)
            The validation samples;
        yval : numpy array of shape (n_validation_samples)
            The validation labels (0:unlabeled, 1:anomaly, -1:normal);
        budget : int
            The budget to allocate;
        rejected : list, optional
            The rejected training instances in the previous allocation round
            Not used in the first round;
        """
        
        chosen = self.check_reward()
        if chosen == "AL":
            self.nroundAL += 1
            ytrain = self.add_training_labels(ytrain, budget, rejected)
            detector = SSIF()
            detector.fit(Xtrain, ytrain)
            
            self.val_probs = detector.predict_proba(Xval)[:,1]
            thresholds = self.set_rejection_thresholds(self.val_probs, yval, seed=self.rng.randint(0,100))

            self.train_probs = detector.predict_proba(Xtrain)[:,1]
            probs = np.concatenate((self.train_probs, self.val_probs))

            self.ALrewards[self.nroundAL], _, self.oldpredAL, _ = self.measure_rewards(probs, thresholds)   
        else:
            detector = SSIF()
            detector.fit(Xtrain, ytrain)
            
            self.train_probs = detector.predict_proba(Xtrain)[:,1]
            self.val_probs = detector.predict_proba(Xval)[:,1]
            
            self.nroundLR += 1
            yval = self.add_validation_labels(yval, self.val_probs, budget)
            thresholds = self.set_rejection_thresholds(self.val_probs, yval, seed=self.rng.randint(0,100))
            probs = np.concatenate((self.train_probs, self.val_probs))
            _, self.LRrewards[self.nroundLR], _, self.oldpredLR = self.measure_rewards(probs, thresholds)
            
        al_scores = 2*np.where(self.train_probs < 0.5, abs(self.train_probs - thresholds[0]), abs(self.train_probs - thresholds[1]))
        self.indicesRanked = np.argsort(al_scores)
        return detector, thresholds
    
    
    def evaluate_model(self, Xtest, ytest, detector, thresholds):
        """
        This function evaluates the algorithm. 
        It computes the average test cost per instance, the false positive rate, the false negative rate, and the rejection rate.
        
        Parameters
        ----------
        Xtest : numpy array of shape (n_samples, n_features)
            The test samples;
        ytest : numpy array of shape (n_samples)
            The test true labels;
        detector : object
            The detector trained in the allocation round;
        thresholds : tuple
            The rejected thresholds optimized in the allocation round;
        """
        test_probs = detector.predict_proba(Xtest)[:,1]
        test_pred = np.where(test_probs >= 0.5, 1, -1)
        reject_test = self.reject(test_probs, thresholds)
        test_pred[reject_test] = 2
        tc, fpr, fnr, rr = self.cost_function(ytest, test_pred)
        return tc, fpr, fnr, rr, test_probs
        

    def initialize_loop(self, X):
        """
        This function initialize the allocation loop by computing the initial rewards. 
        
        Parameters
        ----------
        X : numpy array of shape (n_training_samples+n_validation_samples, n_features)
            The input samples;
        """
        detector = SSIF()
        detector.fit(X, np.zeros((len(X),)))
        
        probs = detector.predict_proba(X)[:,1]
        self.oldpredAL = -np.nan_to_num(probs*np.log2(probs)+(1-probs)*np.log2(1-probs), nan=0.0, posinf=0.0, neginf=0.0)
        rejection_threshold = self.contamination
        reject_probs = np.exp(np.log(.5)*np.power(np.divide(np.array(2*abs(probs - 0.5)), rejection_threshold), 2))
        self.oldpredLR = -np.nan_to_num(reject_probs*np.log2(reject_probs)+(1-reject_probs)*np.log2(1-reject_probs), nan=0.0, posinf=0.0, neginf=0.0)


    def add_validation_labels(self, yval, val_probs, budget):
        """
        This function collects the new validation labels. 
        
        Parameters
        ----------
        yval : numpy array of shape (n_validation_samples)
            The true validation labels;
        val_probs : numpy array of shape (n_validation_samples)
            The anomaly probabilities computed in the allocation round;
        budget : int
            The number of labels to collect;
        """
        for _ in range(budget):
            u = self.rng.uniform(0,1)
            idx = np.argmin(np.where(yval == 0, np.abs(val_probs - u), 10))
            yval[idx] = self.yval[idx]
        return yval

    def set_rejection_thresholds(self, val_probs, yval, seed=9, n_jobs=1):
        """
        This function optimizes the rejection thresholds. 
        
        Parameters
        ----------
        val_probs : numpy array of shape (n_validation_samples)
            The anomaly probabilities computed in the allocation round;
        yval : numpy array of shape (n_validation_samples)
            The validation labels (0:unlabeled, 1:anomaly, -1:normal);
        seed : int, optional
            Seed for the optimization function;
            It is set to 9 by default
        njobs : int, optional
            The number of core to use
            It uses only 1 core by default, use n_jobs=-1 to use all the available cores;
        """
        
        def optimize_cost_function_left(threshold):
            ## using only probabilities lower than 0.5
            
            val_labeled = val_probs[yval != 0]
            labels = yval[yval!= 0]
            
            labels = labels[val_labeled < 0.5]
            val_labeled = val_labeled[val_labeled < 0.5]
            non_rejected = np.where(val_labeled < threshold, 1, 0)
            labels_notrej = labels[non_rejected==1]
            false_negatives = len(labels_notrej[labels_notrej==1])
            rejected = np.where(non_rejected==0)[0]
            cost = (false_negatives*self.c_fn + len(rejected) *self.c_r)/len(yval[val_probs< 0.5])
            return cost
        
        def optimize_cost_function_right(threshold):
            ## using only probabilities higher than 0.5
            
            val_labeled = val_probs[yval != 0]
            labels = yval[yval!= 0]

            labels = labels[val_labeled >= 0.5]
            val_labeled = val_labeled[val_labeled >= 0.5]
            non_rejected = np.where(val_labeled > threshold, 1, 0)
            labels_notrej = labels[non_rejected==1]
            false_positives = len(labels_notrej[labels_notrej==-1])
            rejected = np.where(non_rejected==0)[0]
            cost = (false_positives*self.c_fp + len(rejected) *self.c_r)/len(yval[val_probs>= 0.5])
            return cost

        lower_than = val_probs[val_probs<0.5]
        higher_than = val_probs[val_probs>=0.5]

        max_rejected = int(0.5*len(val_probs))
        if self.c_fn > self.c_fp :
            quantile_left = 1-min(len(lower_than)-1, max_rejected)/len(lower_than)
            if len(lower_than) > 0:
                rt = (0.5+lower_than.max())/2
                res = gp_minimize(func = optimize_cost_function_left, dimensions = [(np.quantile(lower_than,quantile_left),rt)], n_calls = 50,
                                random_state = seed, x0 = [lower_than.max()], n_jobs = n_jobs, verbose = False)
                left_threshold = res.x[0]
                rejected_left = len(lower_than[lower_than >= left_threshold])
                max_rejected -= rejected_left
                quantile_right = min(len(higher_than)-1, max_rejected)/len(higher_than)
            else:
                left_threshold = 0.5
            
            if len(higher_than) > 0:
                lt = (0.5+higher_than.min())/2
                t = np.quantile(higher_than,quantile_right) 
                if t == val_probs.max():
                    t = max(val_probs[val_probs != val_probs.max()])
                res = gp_minimize(func = optimize_cost_function_right, dimensions = [(lt,t)], n_calls = 50,
                                random_state = seed, x0 = [higher_than.min()], n_jobs = n_jobs, verbose = False)
                right_threshold = res.x[0]
            else: 
                right_threshold = 0.5
        else:
            quantile_right = min(len(higher_than)-1, max_rejected)/len(higher_than)
            if len(higher_than) > 0:
                lt = (0.5+higher_than.min())/2
                t = np.quantile(higher_than,quantile_right) 
                if t == val_probs.max():
                    t = max(val_probs[val_probs != val_probs.max()])
                res = gp_minimize(func = optimize_cost_function_right, dimensions = [(lt,t)], n_calls = 30,
                                random_state = seed, x0 = [higher_than.min()], n_jobs = n_jobs, verbose = False)
                right_threshold = res.x[0]

                rejected_right = len(higher_than[higher_than <= right_threshold])
                max_rejected -= rejected_right
                quantile_left = 1-min(len(lower_than)-1, max_rejected)/len(lower_than)
            else: 
                right_threshold = 0.5
            if len(lower_than) > 0:
                rt = (0.5+lower_than.max())/2
                res = gp_minimize(func = optimize_cost_function_left, dimensions = [(np.quantile(lower_than,quantile_left),rt)], n_calls = 30,
                                random_state = seed, x0 = [lower_than.max()], n_jobs = n_jobs, verbose = False)
                left_threshold = res.x[0]
            else:
                left_threshold = 0.5
        return (left_threshold, right_threshold)

    def check_reward(self):
        """
        This function checks which option between AL and LtR has the highest reward. 
        """
        
        ALrew = 0
        LRrew = 0
        idx = self.nroundAL
        step = 0
        while idx >= 0:
            ALrew += self.gamma**step*self.ALrewards[idx]
            step += 1
            idx -= 1
        idx = self.nroundLR
        step = 0
        while idx >= 0:
            LRrew += self.gamma**step*self.LRrewards[idx]
            step += 1
            idx -= 1
        chosen = "AL" if ALrew > LRrew else "LR"

        return chosen

    def measure_rewards(self, probs, thresholds):
        """
        This function computes the reward for the current allocation round. 
        
        Parameters
        ----------
        probs : numpy array of shape (n_training_samples+n_validation_samples)
            The anomaly probabilities computed in the allocation round;
        thresholds : tuple
            The rejected thresholds optimized in the allocation round;
        """
        
        newpred_AL = -np.nan_to_num(probs*np.log2(probs)+(1-probs)*np.log2(1-probs), nan=0.0, posinf=0.0, neginf=0.0)

        reject_probs = np.zeros((len(probs),))
        reject_probs[probs < 0.5] = 1-np.exp(np.log(.5)*np.power(np.divide(np.array(probs[probs < 0.5]),thresholds[0]), 2))
        reject_probs[probs >= 0.5] = 1-np.exp(np.log(.5)*np.power(np.divide(np.array(1-probs[probs >= 0.5]),1-thresholds[1]), 2))

        newpred_LR = -np.nan_to_num(reject_probs*np.log2(reject_probs)+(1-reject_probs)*np.log2(1-reject_probs), nan=0.0, posinf=0.0, neginf=0.0)
        ALreward = np.sum(abs(self.oldpredAL - newpred_AL))/len(newpred_AL)
        LRreward = np.sum(abs(self.oldpredLR - newpred_LR))/len(newpred_LR)

        return ALreward, LRreward, newpred_AL, newpred_LR

    def cost_function(self, labels, predictions):
        """
        This function computes the average test cost per instance, the false positive rate, the false negative rate, and the rejection rate. 
        
        Parameters
        ----------
        labels : numpy array of shape (n_samples)
            The true labels;
        predictions : numpy array of shape (n_samples)
            The predicted labels (1 : anomaly, -1 : normal, 2 : rejected);
        """
        false_positives = np.shape(np.intersect1d(np.where(predictions == 1)[0], np.where(labels == -1)[0]))[0]
        false_negatives = np.shape(np.intersect1d(np.where(predictions == -1)[0], np.where(labels == 1)[0]))[0]
        nrejections = np.shape(np.where(predictions == 2)[0])[0]
        cost = (false_positives*self.c_fp + false_negatives*self.c_fn + nrejections*self.c_r)/len(labels)
        return cost, false_positives/len(labels), false_negatives/len(labels), nrejections/len(labels)

    def reject(self, probs, thresholds):
        """
        This function selects the samples to reject.
        
        Parameters
        ----------
        probs : numpy array of shape (n_samples)
            The anomaly probabilities;
        thresholds : tuple
            The rejected thresholds optimized in the allocation round;
        """
        reject_idx = np.where(np.logical_and(probs >= thresholds[0], probs <= thresholds[1]))[0]
        return reject_idx


    def add_training_labels(self, ytrain, budget, rejected):
        """
        This function collects the new training labels. 
        
        Parameters
        ----------
        ytrain : numpy array of shape (n_training_samples)
            The true training labels;
        budget : int
            The number of labels to collect;
        rejected : list
            The rejected training instances in the previous allocation round (0 : not rejected, 1 : rejected);
        """
        unlabeledIdx = np.where(np.logical_and(rejected == 0, ytrain == 0))[0]
        commonIdxs = np.array([x for x in self.indicesRanked if x in unlabeledIdx], dtype=int)
        indeces = commonIdxs[:budget]
        ytrain[indeces] = self.ytrain[indeces]
        return ytrain
    
def plot_SADAL(X, y, detector, probs, thresholds, idx): 
    """
    This function is used to plot the intermediate results of the SADAL algorithm. 
    
    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The samples to plot;
    y : numpy array of shape (n_samples)
        The true labels of the samples to plot;
    detector : object
        The detector already trained;
    probs : numpy array of shape (n_samples)
        The anomaly probabilities for the samples to plot;
    thresholds : tuple
        The rejected thresholds;
    idx : int
        The round number;
    """   
    predicted = np.where(probs >= 0.5, 1, -1)
    steps = 100
    reject_idx = np.where(np.logical_and(probs >= thresholds[0], probs <= thresholds[1]))[0]
    predicted[reject_idx] = 2

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)

    xmin, xmax = x_min - 0.1 * x_range, x_max + 0.1 * x_range
    ymin, ymax = y_min - 0.1 * y_range, y_max + 0.12 * y_range

    # make the meshgrid based on the data 
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, int(steps)), np.linspace(ymin, ymax, int(steps)))
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    fig = plt.figure(figsize=(9,6), facecolor='w', edgecolor='w')
    
    sh = X_mesh.shape[0]
    pos_probs_background = detector.predict_proba(X_mesh)[:,1]
    
    pos_probs_background = pos_probs_background.reshape(xx.shape)
    plt.contourf(xx, yy, pos_probs_background, cmap=plt.cm.get_cmap('coolwarm'), norm=matplotlib.colors.Normalize(vmin=0, vmax=1), alpha=1)
    
    Z = np.where(pos_probs_background>=thresholds[0],1,0)
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Grays, alpha = 1)
    
    Z = np.where(pos_probs_background>=thresholds[1],1,0)
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Grays, alpha = 1)
    
    reject_probs = np.zeros((pos_probs_background.shape[0],pos_probs_background.shape[1]))
    reject_probs[pos_probs_background < 0.5] = 1-np.exp(np.log(.5)*np.power(np.divide(np.array(pos_probs_background[pos_probs_background < 0.5]),thresholds[0]), 2))
    reject_probs[pos_probs_background >= 0.5] = 1-np.exp(np.log(.5)*np.power(np.divide(np.array(1-pos_probs_background[pos_probs_background >= 0.5]),1-thresholds[1]), 2))

    sh = pos_probs_background.shape[0]*pos_probs_background.shape[1]
    reject_probs = reject_probs.reshape(xx.shape)
    plt.contourf(xx, yy, reject_probs, levels = [0.5, 1], colors = 'gray')


    ax, _ = matplotlib.colorbar.make_axes(plt.gca(), shrink=1.0, pad=0.05)
    cbar = matplotlib.colorbar.ColorbarBase(ax,cmap=plt.cm.get_cmap('coolwarm'),norm=matplotlib.colors.Normalize(vmin=0,vmax=1),
                                     ticks = np.arange(0,105,20)/100)
    plt.subplot(1,1,1)
    NewDtrain = X[[x for x in range(len(X)) if predicted[x]==y[x]]]
    newy = y[[x for x in range(len(X)) if predicted[x]==y[x]]]
    class1pred = NewDtrain[np.where(newy == 1)]
    class0pred = NewDtrain[np.where(newy == -1)]
    cl0 = plt.scatter(class0pred.T[0],class0pred.T[1], 250, c = 'white', alpha = 0.7, edgecolor = 'black',
                      label = 'Class 0')
    plt.scatter(class1pred.T[0],class1pred.T[1], 250, c = 'white', marker = '^',edgecolor = 'black',
                      alpha = 0.7, label = 'Class 1')
    
    rejections = X[[x for x in range(len(X)) if np.logical_and(predicted[x]!=y[x],predicted[x]==2)]]
    newy = y[[x for x in range(len(X)) if np.logical_and(predicted[x]!=y[x],predicted[x]==2)]]
    class1pred = rejections[np.where(newy == 1)]
    class0pred = rejections[np.where(newy == -1)]
    cl2 = plt.scatter(class0pred.T[0],class0pred.T[1], 250, c = 'gray', alpha = 0.7, edgecolor = 'black',
                      label = 'Class 0')
    plt.scatter(class1pred.T[0],class1pred.T[1], 250, c = 'gray', marker = '^',edgecolor = 'black',
                      alpha = 0.7, label = 'Class 1')
    
    mistakes = X[[x for x in range(len(X)) if np.logical_and(predicted[x]!=y[x],predicted[x]!=2)]]
    newy = y[[x for x in range(len(X)) if  np.logical_and(predicted[x]!=y[x],predicted[x]!=2)]]
    class1pred = mistakes[np.where(newy == 1)]
    class0pred = mistakes[np.where(newy == -1)]
    cl1 = plt.scatter(class0pred.T[0],class0pred.T[1], 250, c = 'purple', alpha = 0.7, edgecolor = 'black',
                      label = 'Class 0')
    plt.scatter(class1pred.T[0],class1pred.T[1], 250, c = 'purple', marker = '^',edgecolor = 'black',
                      alpha = 0.7, label = 'Class 1')
    
    title = "Round : " + str(idx)
    plt.title(title, fontsize=15)
    labels = ['Correct', 'Misprediction', 'Rejection']
    colors = ['white', 'purple', 'gray']
    
    handles = [
        Patch(facecolor=color, label=label) 
        for label, color in zip(labels, colors)
    ]  
    plt.legend(handles = handles,
               scatterpoints=1,
               loc='lower center',fontsize=13, bbox_to_anchor=(0.5,0),
               ncol=3,frameon=True).get_frame().set_edgecolor('black')

    plt.axis('off')
    sns.despine(right = True)
    plt.show()
    return
