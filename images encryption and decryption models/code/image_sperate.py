from PIL import Image
import matplotlib.pyplot as plt

# Load image
img = Image.open('im1_all.png')
w, h = img.size

# Split into 4 equal images
sub_w = w // 4
subs = [img.crop((i*sub_w, 0, (i+1)*sub_w, h)) for i in range(4)]

# Save each cropped image
for i, s in enumerate(subs):
    filename = f"img_{i+1}.png"
    s.save(filename)
    print(f"Saved: {filename}")

# Display results
plt.figure(figsize=(12,4))
for i, s in enumerate(subs):
    plt.subplot(1,4,i+1)
    plt.imshow(s)
    plt.axis('off')

plt.show()
