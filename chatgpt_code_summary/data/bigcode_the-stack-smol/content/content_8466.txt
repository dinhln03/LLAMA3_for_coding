textSpeakDictionary = {
    "rs"    : "risos" ,
    "tmb"   : "também" 
}

#imprime o dicionário inteiro
print( "Dicionário =" , textSpeakDictionary )

#imprime apenas o conteúdo relacionado à chave "rs"
print( "\nrs =" , textSpeakDictionary["rs"])

#texto que pede a entrada do usuário
key = input("\nO que você gostaria de converter? : ")
print( key , "=" , textSpeakDictionary[key] )