import matplotlib.pyplot as mpl
import numpy as np
import random, copy

def convert_pace(raw_pace):
	split_pace = raw_pace[:-2].split(":")
	return float(split_pace[0].strip()) + (float(split_pace[1].strip())/60)

def convert_pace_to_seconds(raw_pace):
	split_pace = raw_pace[:-2].split(":")
	return (int(split_pace[0].strip()) * 60) + int(split_pace[1].strip())

def convert_gap(raw_gap):
	split_gap = raw_gap.split(":")
	return float(split_gap[0].strip()) + (float(split_gap[1].strip())/60)

paces = list()
gaps = list()
times = np.zeros(600)
with open("race_results") as f:
	for line in f.readlines():
		if line[-3:].strip() == "/M":
			split_line = line.split()
			pace = convert_pace(split_line[-1])
			gap = convert_gap(split_line[-2])
			pace_in_seconds = convert_pace_to_seconds(split_line[-1])

			if gap < 15:
				paces.append(pace)
				gaps.append(gap)
				if pace_in_seconds < 900:
					times[pace_in_seconds - 300] += 1

smooth_dim = 14
smoothed_times = np.zeros(600 - smooth_dim)

for i in range(int(smooth_dim / 2), 600 - int(smooth_dim / 2)):
	for j in range(-1 * int(smooth_dim / 2), int(smooth_dim / 2) + 1):
		smoothed_times[i - int(smooth_dim / 2)] += times[i - j]/ (smooth_dim + 1)

# declares centers, clusters
MAX_ITER = 30
K = 9
best_clusters = list()
best_centers = list()
best_score = float("inf")

while best_score > 723:

	centers = list()
	clusters = list()
	old_clusters = list()
	for i in range(0, K):
		clusters.append(list())
		centers.append(0)

	# randomly assigns person to clusters
	for i in range(0, len(gaps)):
		clusters[random.randint(0, K-1)].append(i)
		old_clusters.append(-1)

	# finds center of each cluster
	for counter, cluster in enumerate(clusters):
		centers[counter] = 0
		for i in cluster:
			centers[counter] += gaps[i]

		if len(cluster) > 0:
			centers[counter] /= len(cluster)
		
	# runs until no changes are made
	changes = len(gaps)
	while changes > 0:

		# redeclares variables
		changes = 0
		clusters = list()
		for i in range(0, K):
			clusters.append(list())

		for i, gap in enumerate(gaps):
			best_dist = float("inf")
			best_index = -1
			for index, center in enumerate(centers):
				dist = (gap - center)**2

				if dist < best_dist:
					best_dist = dist
					best_index = index

			clusters[best_index].append(i)
			if best_index != old_clusters[i]:
				changes += 1
				old_clusters[i] = best_index

		for i in range(0, K):
			centers[i] = 0

		for i, index in enumerate(old_clusters):
			centers[index] += (gaps[i]/len(clusters[index]))

	# calculates cumulative score
	total_dist = 0
	for i, cluster in enumerate(clusters):
		for j in cluster:
			total_dist += (gaps[j] - centers[i])**2

	# determines if this is the best cluster
	if total_dist < best_score:
		best_score = total_dist
		best_clusters = copy.deepcopy(clusters)
		best_centers = copy.deepcopy(centers)

runner_to_cluster = list()
for i in range(len(gaps)):
	runner_to_cluster.append(-1)
for i, cluster in enumerate(best_clusters):
	for runner in cluster:
		runner_to_cluster[runner] = i

sorted_centers = list(sorted(best_centers))
rearrange_indices = list()
for i in range(0, K):
	rearrange_indices.append(-1)

for i in range(0, K):
	for j in range(0, K):
		if best_centers[j] == sorted_centers[i]:
			rearrange_indices[j] = i

for i in range(0, len(old_clusters)):
	runner_to_cluster[i] = rearrange_indices[runner_to_cluster[i]]


average_pace = list()
std_pace = list()
size_of_groups = list()
pct_compliant = list()
for i in range(0, K):
	average_pace.append(0)
	std_pace.append(0)
	size_of_groups.append(0)
	pct_compliant.append(0)

correct_pace = [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
for i, cluster in enumerate(best_clusters):
	cluster_paces = list()
	for runner in cluster:
		cluster_paces.append(paces[runner])
		if paces[runner] < correct_pace[rearrange_indices[i]]:
			pct_compliant[rearrange_indices[i]] += 1 / len(cluster)
	average_pace[rearrange_indices[i]] = np.mean(cluster_paces)
	std_pace[rearrange_indices[i]] = np.std(cluster_paces)
	size_of_groups[rearrange_indices[i]] = len(cluster_paces)



ticks = ["6:00", "6:30", "7:00", "7:30", "8:00", "8:30", "9:00", "9:30", "10:00"]
#mpl.bar(range(0, K), average_pace, yerr=std_pace, tick_label=ticks, linewidth=0.25)
#mpl.bar(range(0, K), size_of_groups, tick_label=ticks, linewidth=0.25)
#mpl.scatter(gaps, paces, s=1)
#mpl.bar(range(0, K), pct_compliant, tick_label=ticks, linewidth=0.25)
#mpl.bar(range(300 + int(smooth_dim / 2), 900 - int(smooth_dim / 2)), smoothed_times)


mpl.show()





