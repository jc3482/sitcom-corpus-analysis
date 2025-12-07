import re
import pandas as pd
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

def preprocess_text(
    df,
    column_name,
    remove_mbti_words=False
):
    df = df.copy()

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    df[column_name] = df[column_name].astype(str)

    # 1. Remove URLs
    df[column_name] = df[column_name].apply(
        lambda x: re.sub(r'https?:\/\/\S+', '', x.replace("|", " "))
    )

    # 2. Lowercase EARLY
    df[column_name] = df[column_name].str.lower()

    # 3. Remove extremely long nonsense words
    df[column_name] = df[column_name].apply(
        lambda x: re.sub(r'\b\w{30,1000}\b', ' ', x)
    )

    # 4. Remove MBTI labels (optional)
    if remove_mbti_words:
        mbti_types = [
            'infp','infj','intp','intj','entp','enfp','istp','isfp',
            'entj','istj','enfj','isfj','estp','esfp','esfj','estj'
        ]
        pattern = re.compile(r'\b(' + "|".join(mbti_types) + r')\b')
        df[column_name] = df[column_name].apply(lambda x: pattern.sub(' ', x))

    # 5. Remove character names
    names = [
        "Sheldon", "Cooper",
        "Leonard", "Hofstadter",
        "Penny", "Hofstadter",
        "Howard", "Wolowitz",
        "Rajesh", "Koothrappali",
        "Amy", "Farrah", "Fowler",
        "Bernadette", "Rostenkowski", "Wolowitz",

        "Rachel", "Green",
        "Monica", "Geller",
        "Phoebe", "Buffay",
        "Ross", "Geller",
        "Chandler", "Bing",
        "Joey", "Tribbiani",

        "Jay", "Pritchett",
        "Gloria", "Pritchett",
        "Manny", "Delgado",
        "Joe", "Pritchett",
        "Phil", "Dunphy",
        "Claire", "Dunphy",
        "Haley", "Dunphy",
        "Alex", "Dunphy",
        "Luke", "Dunphy",
        "Mitchell", "Pritchett",
        "Cameron", "Tucker",
        "Lily", "Tucker", "Pritchett",

        "Jerry", "Seinfeld",
        "George", "Costanza",
        "Elaine", "Benes",
        "Cosmo", "Kramer",

        "Michael", "Scott",
        "Dwight", "Schrute",
        "Jim", "Halpert",
        "Pam", "Beesly",
        "Ryan", "Howard",
        "Kelly", "Kapoor",
        "Angela", "Martin",
        "Oscar", "Martinez",
        "Kevin", "Malone",
        "Stanley", "Hudson",
        "Phyllis", "Vance",
        "Meredith", "Palmer",
        "Creed", "Bratton",
        "Toby", "Flenderson",
        "Darryl", "Philbin",
        "Andy", "Bernard",
        "Erin", "Hannon",
        "Robert", "California"
    ]
    names = set([n.lower() for n in names])

    df[column_name] = df[column_name].apply(
        lambda text: " ".join([w for w in text.split() if w not in names])
        )

    # 6. Final whitespace normalize
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    return df


def add_mbti_binary_columns(df, type_col="type"):
    """
    Add 4 binary columns:
        EI: 1 = E, 0 = I
        SN: 1 = S, 0 = N
        TF: 1 = T, 0 = F
        JP: 1 = J, 0 = P
    """
    df = df.copy()
    mbti_str = df[type_col].astype(str).str.upper()

    df["EI"] = mbti_str.str[0].map({'E': 1, 'I': 0})
    df["SN"] = mbti_str.str[1].map({'S': 1, 'N': 0})
    df["TF"] = mbti_str.str[2].map({'T': 1, 'F': 0})
    df["JP"] = mbti_str.str[3].map({'J': 1, 'P': 0})

    return df

data = pd.read_csv("mbti_data.csv")

data = preprocess_text(data, column_name="posts", remove_mbti_words=True)
df_encoded = add_mbti_binary_columns(data, type_col="type")
dimension_cols = ["EI", "SN", "TF", "JP"]
texts = df_encoded["posts"].tolist()
labels = df_encoded[dimension_cols].values
types = data["type"]

MODEL_NAME = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
max_len = 500 

class MBTIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.labels[idx]  # shape (4,)

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }

from sklearn.model_selection import train_test_split

indices = np.arange(len(texts))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=types
)

train_indices_final, val_indices = train_test_split(
    train_indices,
    test_size=0.125,
    random_state=42,
    stratify=types.iloc[train_indices] 
)

train_texts = [texts[i] for i in train_indices_final]
val_texts   = [texts[i] for i in val_indices]
test_texts  = [texts[i] for i in test_indices]

train_labels = labels[train_indices_final]
val_labels   = labels[val_indices]
test_labels  = labels[test_indices]

train_dataset = MBTIDataset(train_texts, train_labels, tokenizer, max_len=max_len)
val_dataset   = MBTIDataset(val_texts,   val_labels,   tokenizer, max_len=max_len)
test_dataset  = MBTIDataset(test_texts,  test_labels,  tokenizer, max_len=max_len)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)


class MBTIBertModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=4, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)  # 4 维输出

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)   # batch_size x 4

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return logits, loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = MBTIBertModel(model_name=MODEL_NAME, num_labels=4, dropout=0.2).to(device)

epochs   = 10
lr       = 2e-5
patience = 3
min_delta = 0.0

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        optimizer.zero_grad()

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0.0

    all_labels = []
    all_preds  = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += loss.item()

            probs = torch.sigmoid(logits)     # (batch, 4)
            preds = (probs > 0.5).long()
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds  = torch.cat(all_preds,  dim=0).numpy()

    acc_per_dim = {}
    for i, dim in enumerate(dimension_cols):
        acc_per_dim[dim] = accuracy_score(all_labels[:, i], all_preds[:, i])

    avg_loss = total_loss / len(data_loader)
    return avg_loss, acc_per_dim


best_val_loss = float("inf")
best_state_dict = None
best_val_acc_per_dim = None
patience_counter = 0

for epoch in range(1, epochs + 1):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc_per_dim = eval_epoch(model, val_loader, device)

    print(f"\nEpoch {epoch}/{epochs}")
    print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
    for dim in dimension_cols:
        print(f"  Val accuracy {dim}: {val_acc_per_dim[dim]:.4f}")

    if val_loss + min_delta < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = model.state_dict()
        best_val_acc_per_dim = val_acc_per_dim
        patience_counter = 0
        print("  -> New best model on val set! (saved in memory)")
    else:
        patience_counter += 1
        print(f"  -> No improvement, patience {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
else:
    print("[WARNING] best_state_dict is None, using last-epoch model.")

test_loss, test_acc_per_dim = eval_epoch(model, test_loader, device)
print("\n=== Final Test Performance ===")
print(f"Test loss: {test_loss:.4f}")
for dim in dimension_cols:
    print(f"  Test accuracy {dim}: {test_acc_per_dim[dim]:.4f}")

save_path = "mbti_bundle.pkl"

bundle = {
    "tokenizer_name": MODEL_NAME,
    "max_len": max_len,
    "state_dict": model.state_dict(),
    "label_mapping": {
        "EI": {1: "E", 0: "I"},
        "SN": {1: "S", 0: "N"},
        "TF": {1: "T", 0: "F"},
        "JP": {1: "J", 0: "P"},
    },
    "preprocessing_info": {
        "column_name": "posts",
        "remove_mbti_words": True,
        "remove_names": True,
    },
    "val_metrics": {
        "best_val_loss": best_val_loss,
        "accuracy_per_dim": best_val_acc_per_dim,
    },
    "test_metrics": {
        "test_loss": test_loss,
        "accuracy_per_dim": test_acc_per_dim,
    }
}

joblib.dump(bundle, save_path)
print("Saved RoBERTa MBTI bundle to", save_path)
