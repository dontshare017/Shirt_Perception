import pandas as pd
import matplotlib.pyplot as plt
import json

with open('/Users/YifeiHu/Downloads/json.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data, columns=['wall_time', 'epoch', 'loss'])
df['loss'] = df['loss'] / 16

plt.figure(figsize=(5, 5))
plt.plot(df['epoch'], df['loss'], label='MSE Loss per Image')

plt.title('Validation MSE Loss per Image')
plt.xlabel('Epoch')
plt.ylabel('Loss (Pixel Coordinate Squared)')
plt.ylim(200, 700)
plt.legend()
plt.grid(True)
plt.show()
