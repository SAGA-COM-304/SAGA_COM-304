import numpy as np
from pathlib import Path
import argparse
from collections import Counter

def analyze_vocab_size(token_dir: Path, num_files: int = 50):
    """
    Analyse la taille du vocabulaire utilisé dans les tokens audio.
    
    Args:
        token_dir: Répertoire contenant les fichiers .npy
        num_files: Nombre de fichiers à analyser
    """
    print(f"🔍 Analyse du vocabulaire des tokens audio")
    print(f"📁 Répertoire: {token_dir}")
    print(f"📊 Analysing {num_files} fichiers...\n")
    
    # Collecter tous les tokens
    all_tokens = []
    shapes = []
    
    token_files = list(token_dir.glob("*.npy"))[:num_files]
    
    if not token_files:
        print(f"❌ Aucun fichier .npy trouvé dans {token_dir}")
        return
    
    for i, token_file in enumerate(token_files):
        try:
            tokens = np.load(token_file)
            all_tokens.append(tokens)
            shapes.append(tokens.shape)
            
            if i % 10 == 0:
                print(f"  Traitement: {i+1}/{len(token_files)} fichiers...")
                
        except Exception as e:
            print(f"❌ Erreur avec {token_file.name}: {e}")
    
    if not all_tokens:
        print("❌ Aucun token valide trouvé!")
        return
    
    # Analyser par codebook (colonne)
    print(f"\n{'='*70}")
    print("📊 ANALYSE PAR CODEBOOK")
    print(f"{'='*70}")
    
    # Déterminer le nombre de codebooks
    num_codebooks = all_tokens[0].shape[1]
    print(f"🔢 Nombre de codebooks: {num_codebooks}")
    
    vocab_sizes_per_codebook = []
    
    for codebook_idx in range(num_codebooks):
        # Collecter tous les tokens pour ce codebook
        codebook_tokens = []
        for tokens in all_tokens:
            codebook_tokens.extend(tokens[:, codebook_idx].flatten())
        
        # Calculer les statistiques
        unique_tokens = np.unique(codebook_tokens)
        vocab_size = len(unique_tokens)
        min_token = min(codebook_tokens)
        max_token = max(codebook_tokens)
        
        vocab_sizes_per_codebook.append(vocab_size)
        
        print(f"  Codebook {codebook_idx:2d}: "
              f"Vocab={vocab_size:4d}, Range=[{min_token:4d}, {max_token:4d}]")
        
        # Montrer la distribution pour les premiers codebooks
        if codebook_idx < 3:
            counter = Counter(codebook_tokens)
            most_common = counter.most_common(5)
            print(f"    Top 5 tokens: {most_common}")
    
    # Résumé global
    print(f"\n{'='*70}")
    print("📈 RÉSUMÉ GLOBAL")
    print(f"{'='*70}")
    
    print(f"🔢 Vocab size par codebook:")
    print(f"  Min: {min(vocab_sizes_per_codebook)}")
    print(f"  Max: {max(vocab_sizes_per_codebook)}")
    print(f"  Moyenne: {np.mean(vocab_sizes_per_codebook):.1f}")
    
    # Vérifier la cohérence
    all_vocab_same = len(set(vocab_sizes_per_codebook)) == 1
    print(f"🎯 Vocabulaire uniforme: {'✅ Oui' if all_vocab_same else '❌ Non'}")
    
    if all_vocab_same:
        vocab_size = vocab_sizes_per_codebook[0]
        print(f"🎉 Taille du vocabulaire: {vocab_size}")
        
        # Vérifier si c'est une puissance de 2
        import math
        if vocab_size > 0 and (vocab_size & (vocab_size - 1)) == 0:
            bits = int(math.log2(vocab_size))
            print(f"💡 C'est 2^{bits} (puissance de 2)")
    
    # Analyse additionnelle
    print(f"\n📋 INFORMATIONS ADDITIONNELLES:")
    
    # Calculer la densité d'utilisation
    theoretical_max = 2048  # Vocab size théorique de Mimi
    if all_vocab_same and vocab_sizes_per_codebook[0] <= theoretical_max:
        usage_density = vocab_sizes_per_codebook[0] / theoretical_max * 100
        print(f"📊 Densité d'utilisation du vocab théorique: {usage_density:.1f}%")
    
    # Analyser les shapes
    unique_shapes = list(set(shapes))
    print(f"🔳 Shapes trouvées: {unique_shapes}")
    
    # Estimer l'efficacité de compression
    total_tokens = sum(tokens.size for tokens in all_tokens)
    total_files = len(all_tokens)
    avg_tokens_per_file = total_tokens / total_files
    print(f"📏 Tokens moyens par fichier: {avg_tokens_per_file:.0f}")

def check_specific_vocab_properties(token_path: Path):
    """
    Vérifications spécifiques pour Mimi.
    """
    print(f"🔬 Vérification des propriétés Mimi pour: {token_path.name}")
    
    tokens = np.load(token_path)
    
    # Vérifications Mimi
    expected_codebooks = 32
    expected_vocab_size = 2048
    expected_range = (0, 2047)
    
    print(f"📐 Shape: {tokens.shape}")
    print(f"🔍 Vérifications Mimi:")
    
    # Nombre de codebooks
    actual_codebooks = tokens.shape[1] if tokens.ndim == 2 else 1
    print(f"  Codebooks: {actual_codebooks} {'✅' if actual_codebooks == expected_codebooks else '❌'} "
          f"(attendu: {expected_codebooks})")
    
    # Range des valeurs
    min_val, max_val = tokens.min(), tokens.max()
    range_ok = min_val >= expected_range[0] and max_val <= expected_range[1]
    print(f"  Range: [{min_val}, {max_val}] {'✅' if range_ok else '❌'} "
          f"(attendu: {expected_range})")
    
    # Vocab size utilisé
    unique_tokens = len(np.unique(tokens))
    print(f"  Tokens uniques: {unique_tokens} {'✅' if unique_tokens <= expected_vocab_size else '❌'} "
          f"(max: {expected_vocab_size})")

def main():
    parser = argparse.ArgumentParser("Analyser la vocab size des tokens audio")
    parser.add_argument("token_path", type=Path,
                       help="Chemin vers le répertoire ou fichier token")
    parser.add_argument("--num_files", type=int, default=50,
                       help="Nombre de fichiers à analyser")
    parser.add_argument("--quick_check", action="store_true",
                       help="Vérification rapide d'un seul fichier")
    
    args = parser.parse_args()
    
    if args.token_path.is_file():
        # Analyse d'un seul fichier
        check_specific_vocab_properties(args.token_path)
    elif args.token_path.is_dir():
        # Analyse complète du répertoire
        analyze_vocab_size(args.token_path, args.num_files)
        
        # Vérification rapide d'un échantillon
        sample_file = next(args.token_path.glob("*.npy"), None)
        if sample_file:
            print(f"\n{'='*70}")
            check_specific_vocab_properties(sample_file)
    else:
        print(f"❌ Chemin non valide: {args.token_path}")

if __name__ == "__main__":
    main()