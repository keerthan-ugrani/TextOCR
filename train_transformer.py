import os
import torch
import pandas as pd
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import Levenshtein as lev
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import time  
from tqdm import tqdm
all_train_losses, all_train_cer, all_train_wer, all_train_times, all_train_bleu = [], [], [], [], []
all_val_losses, all_val_cer, all_val_wer, all_val_times, all_val_bleu = [], [], [], [], []
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 4
NUM_WORKERS = 2
MAX_IMAGE_SIZE = (64, 64)
# please replace it with respective paths 
CSV_PATH = "C:\\Users\\ramsh\\Downloads\\cleaned_annotations.csv\\updated_cleaned_annotations.csv"
IMG_DIR = "C:\\Users\\ramsh\\Documents\\TEXTOCR\\Train\\Train"
CHECKPOINT_PATH = "C:\\Users\\ramsh\\Documents\\TEXTOCR\\output\\checkpoint.pth.tar"
class TextRecognitionDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self.df['text'].astype(str)
        self.characters = sorted(set(char for label in self.labels for char in label))
        self.label_to_int = {char: idx for idx, char in enumerate(self.characters)}
        self.int_to_label = {idx: char for char, idx in self.label_to_int.items()}

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 3])
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"File not found: {img_name}. Using a default image.")
            image = Image.new('RGB', (64, 64), color = 'white')  # replace with your actual default image

        label = self.labels[idx]
        label_encoded = [self.label_to_int[char] for char in label]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_encoded)


class OCRTransformer(nn.Module):
    def __init__(self, num_classes):
        super(OCRTransformer, self).__init__()
        self.embedding = nn.Embedding(num_classes, 512)
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, tgt):
        x = self.resnet(x)
        x = x.unsqueeze(0) 
        tgt = self.embedding(tgt)
        tgt = tgt.permute(1, 0, 2)
        output = self.transformer(x, tgt)
        output = self.fc(output)
        return output


transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),
])

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    max_length = max(len(label) for label in labels)
    labels_padded = torch.zeros(len(labels), max_length, dtype=torch.long)
    for i, label in enumerate(labels):
        labels_padded[i, :len(label)] = label
    return images, labels_padded

class OCRTransformer(nn.Module):
    def __init__(self, num_classes):
        super(OCRTransformer, self).__init__()
        self.embedding = nn.Embedding(num_classes, 512)
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, tgt):
        x = self.resnet(x)
        x = x.unsqueeze(0) 
        tgt = self.embedding(tgt)
        tgt = tgt.permute(1, 0, 2)
        output = self.transformer(x, tgt)
        output = self.fc(output)
        return output

# Global lists to store metrics for plotting


def plot_metrics(epoch):
    if epoch == 0 or len(all_train_cer) != len(all_val_cer):
        print("Skipping plotting for epoch", epoch)
        return

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, epoch+1), all_train_cer, label='Train CER')
    plt.plot(range(1, epoch+1), all_val_cer, label='Val CER')
    plt.xlabel('Epoch')
    plt.ylabel('CER')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(1, epoch+1), all_train_wer, label='Train WER')
    plt.plot(range(1, epoch+1), all_val_wer, label='Val WER')
    plt.xlabel('Epoch')
    plt.ylabel('WER')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(1, epoch+1), all_train_bleu, label='Train BLEU')
    plt.plot(range(1, epoch+1), all_val_bleu, label='Val BLEU')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def compute_metrics(targets, outputs, idx_to_label):
    output_labels = []
    target_labels = []

    for target, output in zip(targets, outputs.argmax(dim=2)):
        target_label = ''.join(idx_to_label[int(idx)] for idx in target if idx != -1)
        output_label = ''.join(idx_to_label[int(idx)] for idx in output if idx != -1)
        target_labels.append(target_label)
        output_labels.append(output_label)

    cer = sum(lev.distance(t, o) for t, o in zip(target_labels, output_labels)) / sum(len(t) for t in target_labels)
    wer = sum(lev.distance(t.split(), o.split()) for t, o in zip(target_labels, output_labels)) / sum(len(t.split()) for t in target_labels)
    smoothie = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([t], o, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie) for t, o in zip(target_labels, output_labels)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return {'cer': cer, 'wer': wer, 'bleu': avg_bleu}

def train_one_epoch(model, train_loader, optimizer, criterion, device, idx_to_label, epoch, scaler):
    model.train()
    total_loss = 0.0
    total_time = 0
    total_bleu = 0
    total_cer = 0  # Added total_cer for Character Error Rate
    total_wer = 0  # Added total_wer for Word Error Rate
    num_batches = len(train_loader)
    smoothie = SmoothingFunction().method4
    
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        start_time = time.time()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            images, labels = images.to(device), labels.to(device)
            tgt_input = labels[:, :-1]
            outputs = model(images, tgt_input)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels[:, 1:].contiguous().view(-1))

        scaler.scale(loss).backward()

        # Clip the gradients to avoid explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_time += time.time() - start_time

        metrics = compute_metrics(labels[:, 1:], outputs, idx_to_label)
        total_cer += metrics['cer']
        total_wer += metrics['wer']
        total_bleu += metrics['bleu']

    avg_loss = total_loss / num_batches
    avg_time = total_time / num_batches
    avg_bleu = total_bleu / num_batches
    avg_cer = total_cer / num_batches  # Calculating average CER
    avg_wer = total_wer / num_batches  # Calculating average WER

    return avg_loss, avg_time, avg_bleu, avg_cer, avg_wer  # Returning all the calculated averages



def evaluate(model, val_loader, criterion, device, idx_to_label, epoch, scaler):
    model.eval()
    total_loss = 0.0
    total_bleu = 0
    total_cer = 0   # Initialize total CER
    total_wer = 0   # Initialize total WER
    total_time = 0
    num_batches = len(val_loader)
    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
            start_time = time.time()
            images, labels = images.to(device), labels.to(device)
            tgt_input = labels[:, :-1]

            with torch.cuda.amp.autocast():
                outputs = model(images, tgt_input)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels[:, 1:].contiguous().view(-1))

            total_loss += loss.item()

            metrics = compute_metrics(labels[:, 1:], outputs, idx_to_label)
            total_cer += metrics['cer']  # Update total CER
            total_wer += metrics['wer']  # Update total WER

            output_labels = [''.join(idx_to_label[int(idx)] for idx in output) for output in outputs.argmax(dim=2).cpu().numpy()]
            target_labels = [''.join(idx_to_label[int(idx)] for idx in target) for target in labels[:, 1:].cpu().numpy()]
            bleu_scores = [sentence_bleu([t], o, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie) for t, o in zip(target_labels, output_labels)]
            total_bleu += sum(bleu_scores) / len(bleu_scores)

            total_time += time.time() - start_time

    avg_loss = total_loss / num_batches
    avg_bleu = total_bleu / num_batches
    avg_cer = total_cer / num_batches  # Calculate average CER
    avg_wer = total_wer / num_batches  # Calculate average WER
    avg_time = total_time / num_batches

    return avg_loss, avg_time, avg_bleu, avg_cer, avg_wer



def plot_loss_and_time(losses, times, title):
    epochs = range(1, len(losses) + 1)
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Time (s)', color=color)  
    ax2.plot(epochs, times, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title(title)
    plt.show()

def plot_single_metric(epochs, metrics, ylabel, title):
    plt.figure(figsize=(5, 4))

    plt.plot(epochs, metrics)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()
def save_checkpoint(state, filename=CHECKPOINT_PATH):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename=CHECKPOINT_PATH):
    print("=> Loading checkpoint")
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    train_metrics = checkpoint.get('train_metrics', None)
    val_metrics = checkpoint.get('val_metrics', None)

    return model, optimizer, checkpoint['epoch'], train_metrics, val_metrics

def plot_bleu_scores(epochs, train_bleu_scores, val_bleu_scores):
    plt.figure(figsize=(10, 5))
    
    plt.plot(epochs, train_bleu_scores, label='Training BLEU Score', marker='o', linestyle='-')
    plt.plot(epochs, val_bleu_scores, label='Validation BLEU Score', marker='o', linestyle='-')
    
    plt.xlabel('Epochs')
    plt.ylabel('BLEU Score')
    plt.title('Training and Validation BLEU Scores')
    plt.legend()
    
    plt.grid(True)
    plt.show()

def load_checkpoint(model, optimizer, filename=CHECKPOINT_PATH):
    print("=> Loading checkpoint")
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None:  # Add this condition to check if optimizer is not None before loading state
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    train_metrics = checkpoint.get('train_metrics', None)
    val_metrics = checkpoint.get('val_metrics', None)

    return model, optimizer, checkpoint['epoch'], train_metrics, val_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor()
    ])
    
    dataset = TextRecognitionDataset(CSV_PATH, IMG_DIR, transform=transform)
    train_size = int(0.4592 * len(dataset))  
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    num_classes = len(dataset.label_to_int)
    model = OCRTransformer(num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Training and validation metrics

    
    start_epoch = 0

     # If a checkpoint exists, load it
    if os.path.isfile(CHECKPOINT_PATH):
        model, optimizer, start_epoch, train_metrics, val_metrics = load_checkpoint(model, optimizer, CHECKPOINT_PATH)
        all_train_losses, all_train_times, all_train_cer, all_train_wer, all_train_bleu = train_metrics
        all_val_losses, all_val_times, all_val_cer, all_val_wer, all_val_bleu = val_metrics
    else:
        all_train_losses, all_train_times, all_train_cer, all_train_wer, all_train_bleu = [], [], [], [], []
        all_val_losses, all_val_times, all_val_cer, all_val_wer, all_val_bleu = [], [], [], [], []

    epoch = start_epoch
    # Training loop
    for epoch in range(start_epoch, 10):  # Adjust the total number of epochs as needed
        start_time = time.time()

        # Training step
        train_loss,train_time,trainBLEU, trainCER, trainWER  = train_one_epoch(model, train_loader, optimizer, criterion, device, dataset.int_to_label, epoch, scaler)
        end_time = time.time() - start_time

        # Store metrics
        all_train_losses.append(train_loss)
        all_train_cer.append(trainCER)
        all_train_wer.append(trainWER)
        all_train_bleu.append(trainBLEU)
        all_train_times.append(train_time)

        print(f"Epoch {epoch+1} - Time: {end_time:.2f}s, Train Loss: {train_loss:.4f}")

    # Calculate and print average training metrics
    avg_train_loss = sum(all_train_losses) / len(all_train_losses)
    avg_train_cer = sum(all_train_cer) / len(all_train_cer)
    avg_train_wer = sum(all_train_wer) / len(all_train_wer)
    avg_train_bleu = sum(all_train_bleu) / len(all_train_bleu)
    avg_train_time = sum(all_train_times) / len(all_train_times)

    print("\nTraining completed. Averages:")
    print(f"Average Train Loss: {avg_train_loss:.4f}")
    print(f"Average Train CER: {avg_train_cer:.4f}")
    print(f"Average Train WER: {avg_train_wer:.4f}")
    print(f"Average Train BLEU: {avg_train_bleu:.4f}")
    print(f"Average Time per Epoch: {avg_train_time:.2f}s")

    # Validation loop
    for epoch in range(start_epoch, 10):  # Adjust the total number of epochs as needed for validation
        start_time = time.time()

        # Validation step
        val_loss,val_time,valBLEU, valCER, valWER = evaluate(model, val_loader, criterion, device, dataset.int_to_label, epoch, scaler)
        end_time = time.time() - start_time

        # Store metrics
        all_val_losses.append(val_loss)
        all_val_cer.append(valCER)
        all_val_wer.append(valWER)
        all_val_bleu.append(valBLEU)
        all_val_times.append(val_time)

        print(f"Validation Epoch {epoch+1} - Time: {end_time:.2f}s, Validation Loss: {val_loss:.4f}")

    # Calculate and print average validation metrics
    avg_val_loss = sum(all_val_losses) / len(all_val_losses)
    avg_val_cer = sum(all_val_cer) / len(all_val_cer)
    avg_val_wer = sum(all_val_wer) / len(all_val_wer)
    avg_val_bleu = sum(all_val_bleu) / len(all_val_bleu)
    avg_val_time = sum(all_val_times) / len(all_val_times)

    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_metrics': (all_train_losses, all_train_times, all_train_cer, all_train_wer, all_train_bleu),
            'val_metrics': (all_val_losses, all_val_times, all_val_cer, all_val_wer, all_val_bleu)
    }, CHECKPOINT_PATH)

    print("\nValidation completed. Averages:")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Validation CER: {avg_val_cer:.4f}")
    print(f"Average Validation WER: {avg_val_wer:.4f}")
    print(f"Average Validation BLEU: {avg_val_bleu:.4f}")
    print(f"Average Time per Epoch: {avg_val_time:.2f}s")


    epochs = range(1, len(all_train_losses) + 1)  # Adjust depending on the total number of epochs
    plot_single_metric(epochs, all_train_losses, 'Loss', 'Training Loss')
    plot_single_metric(epochs, all_train_times, 'Time (s)', 'Training Time')
    plot_single_metric(epochs, all_train_bleu, 'BLEU Score', 'Training BLEU Score')

    plot_single_metric(epochs, all_val_losses, 'Loss', 'Validation Loss')
    plot_single_metric(epochs, all_val_times, 'Time (s)', 'Validation Time')
    plot_single_metric(epochs, all_val_bleu, 'BLEU Score', 'Validation BLEU Score')

    epochs = range(1, len(all_train_bleu) + 1)
    plot_bleu_scores(epochs, all_train_bleu, all_val_bleu)


if __name__ == "__main__":
    main()













































