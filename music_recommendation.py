import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv("SpotifyFeatures.csv")
data = data.sample(10000, random_state=42).reset_index(drop=True)

features = data[['danceability', 'energy', 'tempo', 'loudness', 'valence']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

model = NearestNeighbors(n_neighbors=6, metric='cosine')
model.fit(scaled_features)

def recommend(song_name):
    index = data[data['track_name'].str.lower() == song_name.lower()].index
    if len(index) == 0:
        print("Song not found in dataset")
        return
    index = index[0]
    distances, indices = model.kneighbors([scaled_features[index]])
    print("\nRecommended Songs:")
    for i in indices[0][1:]:
        print(data.iloc[i].track_name)

song = input("Enter a song name: ")
recommend(song)
