from data.DataLoading import get_df
import numpy as np


def get_beliefs(model_conditional_probabilities, model_results):
    """
    This function will return the beliefs for the given inputs. Takes the conditional probabilities from the models and
    the predicitons of the models and calculate the beliefs.
    :param model_conditional_probabilities:
    :param model_results:
    :return:
    """

    n_models = 4

    beliefs = []
    for model in model_results: #For each of the input rows (one plant)
        masses = [[0,0,0] for _ in range(n_models)]
        for i in range(n_models): #get the massss of the individual models
            masses[i][model[i]] = model_conditional_probabilities[i][model[i]]
            masses[i][2] = 1 -model_conditional_probabilities[i][model[i]]
            pass

        for i in range((n_models-1)):
            #In this loop we calculate the next mass Thiss loop takes the last element and the i and combies the masses
            #Then the new massses are appended and combined with the second element, when all elementss are combined
            #The last element of the list contains the masses considering all models
            massA = (masses[i][0] * masses[-1][0] + masses[i][0] * masses[-1][2]+ masses[i][2] * masses[-1][0]) / (1 - (masses[i][0] * masses[-1][1] + masses[i][1] * masses[-1][0]))
            massB = (masses[i][1] * masses[-1][1] + masses[i][1] * masses[-1][2]+ masses[i][2] * masses[-1][1]) / (1 - (masses[i][1] * masses[-1][0] + masses[i][0] * masses[-1][1]))
            massAUB = 1 - (massA + massB)
            masses.append([massA, massB, massAUB])
        beliefs.append(masses[-1])

    return beliefs


def main():
    df = get_df("../data/seedling_labels_with_features.csv")
    mass = {}

    model_conditional_probabilities = [[0.881, 0.962],
                                       [0.910, 0.939],
                                       [0.836, 0.939],
                                       [0.776, 0.917]]
    model_results = [[1, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 1],
       [1, 1, 0, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 1],
       [1, 1, 1, 1],
       [0, 0, 1, 1],
       [0, 0, 1, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 0],
       [1, 1, 1, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 0, 1, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 1],
       [0, 0, 0, 0],
       [1, 0, 0, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 0, 0, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 0, 0, 1],
       [1, 1, 1, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 1, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 1, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 1, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0],
       [1, 1, 1, 1],
       [0, 1, 1, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 0, 0],
       [0, 0, 0, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 0, 1],
       [0, 0, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 0, 0, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 0, 1],
       [0, 0, 1, 0],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 0, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]]
    model_results = np.genfromtxt("models_predictions.csv", delimiter=',').astype(int)

    true_results = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
       0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
       0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
       1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
       0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1,
       0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
       0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
       1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1,
       0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
       1]
    true_results = np.genfromtxt("true_answer.csv", delimiter=',').astype(int)

    beliefs = get_beliefs(model_conditional_probabilities, model_results)
    belies_plausibility = [[belief[0],belief[1],belief[2], (belief[0]+belief[2]), (belief[1]+belief[2]) ] for belief in beliefs]

    belief_binary = []
    for idx, result in enumerate(true_results):
        print(f"The models return: {model_results[idx][0]}, {model_results[idx][1]}, {model_results[idx][2]}, {model_results[idx][3]}                                     Belief 0 Plausib. 0    Belief 1 Plausib. 1")
        print(f"The expert opinion is: {result}, the delta (belief <-> plausibility) is: {belies_plausibility[idx][0]:.3f} <-> {belies_plausibility[idx][3]:.3f},       {belies_plausibility[idx][1]:.3f} <-> {belies_plausibility[idx][4]:.3f}")
        if beliefs[idx][1] > 0.5:
            belief_binary.append(1)
        else:
           belief_binary.append(0)

    correct = [0, 0, 0, 0, 0]
    for idx, result in enumerate(true_results):
       if model_results[idx][0] == true_results[idx]:
          correct[0] += 1
       if model_results[idx][1] == true_results[idx]:
          correct[1] += 1
       if model_results[idx][2] == true_results[idx]:
          correct[2] += 1
       if model_results[idx][3] == true_results[idx]:
          correct[3] += 1
       if belief_binary[idx] == true_results[idx]:
          correct[4] += 1

    print(f"Model 1 has an accuracy of: {correct[0]/len(true_results):.3f}")
    print(f"Model 2 has an accuracy of: {correct[1]/len(true_results):.3f}")
    print(f"Model 3 has an accuracy of: {correct[2]/len(true_results):.3f}")
    print(f"Model 4 has an accuracy of: {correct[3]/len(true_results):.3f}")
    print(f"Dempster-Shafer voting has an accuracy of: {correct[4]/len(true_results):.3f}")


if __name__ == "__main__":

    main()

