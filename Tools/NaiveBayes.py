import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



class NaiveBayes:
    def __init__(self):
        pass
    def fit(self,X,y):
        self.probabilityTabels = {}
        if isinstance(X,dict):
            for key in X.keys():
                self.probabilityTabels[key] = self.__calcualtePriorMarginal(X[key],y,Feature=str(key),show=True)
        elif isinstance(X,list) and isinstance(X[0],list):
            for idx,feature in enumerate(X):
                self.probabilityTabels[idx] = self.__calcualtePriorMarginal(feature, y, Feature=str(idx), show=True)
        else:
            print("Unsupported data in X")


    def probability(self,P):
        '''{"Y":"Male","X":{"Magaine":"Yes","Watch":"Yes","LifeInsurance":"No","CreditCard":"Yes"}}'''

        nominator = 1
        for key in P["X"].keys():
            z = self.probabilityTabels[key][f"Y={P['Y']}"][f"X={P['X'][key]}"]
            n = self.probabilityTabels[key][f"Y={P['Y']}"][f"P(Y)"]
            nominator *= (z/n)
            print(f"""{f"P({key}={P['X'][key]} | Y={P['Y']})":30s}""" + f" = {z:.2f}/{n:.2f} -> {z/n:.4f}")

        nominator *= self.probabilityTabels[key][f"Y={P['Y']}"][f"P(Y)"]
        print()

        denominotor = 1
        for key in P["X"].keys():
            d = self.probabilityTabels[key][f"P(X)"][f"X={P['X'][key]}"]
            denominotor *= d
            print(f"""{f"P({key}={P['X'][key]})":30s}""" + f" = {d:.2f}")
        print()

        pString = f"P(Y={P['Y']}| "
        for key in P["X"].keys():
            pString += f"{key}={P['X'][key]}, "
        pString = pString[:-2]+")"
        print(f"{pString} = {nominator/denominotor:.5f} -> {(nominator/denominotor)*100:.2f}%")


    def __calcualtePriorMarginal(self,X, Y, Feature="Magazine", show=False):
        X_lbl = list(set(X))
        Y_lbl = list(set(Y))

        prob = np.zeros((len(X_lbl) + 1, len(Y_lbl) + 1))
        for entity, value in zip(X, Y):
            idx_X = X_lbl.index(entity)
            idx_Y = Y_lbl.index(value)
            prob[idx_X][idx_Y] += 1
        prob /= len(X)

        for idx_X in range(0, len(X_lbl)):
            for idx_Y in range(0, len(Y_lbl)):
                prob[idx_X][len(Y_lbl)] += prob[idx_X][idx_Y]
                prob[len(X_lbl)][idx_Y] += prob[idx_X][idx_Y]

        prob[-1][-1] = 1

        X_lbl = ["X=" + lbl for lbl in X_lbl]
        Y_lbl = ["Y=" + lbl for lbl in Y_lbl]

        X_lbl.append("P(Y)")
        Y_lbl.append("P(X)")

        df = pd.DataFrame(prob, index=X_lbl,
                          columns=Y_lbl)
        if show:
            plt.title(f"Probability {Feature}")
            sns.heatmap(df, annot=True)
            plt.show()

        return df

if __name__ == '__main__':
    Y = ["Male", "Male", "Male", "Male", "Male", "Male", "Female", "Female", "Female", "Female"]  # Label

    X = {"Magaine": ["No", "No", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes"],
         "Watch": ["Yes", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "Yes"],
         "LifeInsurance": ["No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "Yes", "Yes"],
         "CreditCard": ["No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes"]}  # Features

    P = {"Y": "Male", "X": {"Magaine": "Yes", "Watch": "Yes", "LifeInsurance": "No", "CreditCard": "Yes"}}

    clf = NaiveBayes()
    clf.fit(X, Y)
    clf.probability(P)