with open('logistics_dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Total characters in dataset : {len(text):,}")
print(f"Total unique characters      : {vocab_size}")
print(f"Character set                : {repr(''.join(chars))}")
print()
print("--- Character breakdown ---")
for ch in chars:
    if ch == '\n':
        name = "NEWLINE"
    elif ch == ' ':
        name = "SPACE"
    elif ch == '\t':
        name = "TAB"
    else:
        name = ch
    print(f"  '{name}'  (ord={ord(ch)})")
