import sys

def main():
    if len(sys.argv) != 4:
        print("Erreur : Nombre d'arguments incorrect.")
        return
    
    name = sys.argv[1]
    age = sys.argv[2]
    smoker = sys.argv[3]

    # Traitement des données
    response = f"Patient : {name}, Age : {age}, Fumeur : {'Oui' if smoker == 'true' else 'Non'}"
    
    #Si on print, une boite de dialogue s'afficiherait avec le message 
    print(response)

if __name__ == "__main__":
    main()
