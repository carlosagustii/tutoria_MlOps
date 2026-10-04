# UBICACION FINAL (ejercicio 1): src/claims_triage/cli.py
"""Programa principal: CSV -> validacion -> preprocess -> modelo -> CSV de salida.

Se ejecuta asi (desde la raiz del proyecto):
    uv run python -m claims_triage.cli --input data/claims.csv --output .tmp/predicciones.csv

Pasos del programa:
  1. read_claims : lee el CSV y valida cada fila con ClaimRequest    (ejercicio 2)
  2. load_model  : carga el modelo                                    (ejercicio 4)
  3. predict     : predice cada siniestro                             (ejercicios 3 y 4)
  4. escribe el CSV de salida                                         (ejercicio 4)
"""

import argparse
import csv  # noqa: F401
import os  # noqa: F401
import sys  # noqa: F401

from pydantic import ValidationError  # noqa: F401

from claims_triage.contracts import ClaimRequest  # noqa: F401
from claims_triage.inference import load_model, predict  # noqa: F401

# Columnas del CSV de salida, en este orden.
OUTPUT_COLUMNS = ["claim_id", "decision", "risk_probability", "model_version"]


def read_claims(path):
    """EJERCICIO 2: leer el CSV y devolver la lista de siniestros validados."""
    # TODO 2.8: leer el fichero CSV indicado, cuyas filas tienen las columnas
    #   de ClaimRequest.
    lista =[]
    with open(path, "r", encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f)
        
    # TODO 2.9: validar cada fila con ClaimRequest. Si alguna fila no es valida,
    #   hay que parar con un ValueError cuyo mensaje diga en que LINEA del
    #   fichero esta el problema y cual es (la cabecera es la linea 1).
    
        for numero, fila in enumerate(lector, start=2): 
            #la cabecera esta en la linea 1 asi que empezamos por 2
            try:
                #sin los astericos no funciona, porque ClaimRequest espera
                #argumentos con nombre, y fila es un diccionario tipo {nombre_columna: valor, ...}
                lista.append(ClaimRequest(**fila)) 
                
            except ValidationError as e:
                raise ValueError(f"Error {e} en la linea {numero}")


    # TODO 2.10: devolver la lista con todos los siniestros validados.
    return lista


def main(argv=None):
    # Ya hecho: lee los argumentos --input, --output y --model.
    parser = argparse.ArgumentParser(description="Triaje de siniestros")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/claims_triage_model.joblib")
    args = parser.parse_args(argv)  # noqa: F841

    # TODO 4.12: leer y validar los siniestros y cargar el modelo. Si algo de
    #   eso falla (ValueError), mostrar el error por la salida de errores y
    #   terminar con codigo de salida 2, SIN escribir ningun fichero.
    try:
        siniestros = read_claims(args.input)
        modelo = load_model(args.model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    # TODO 4.13: obtener la prediccion de cada siniestro.
    predicciones = []
    for siniestro in siniestros:
        predicciones.append(predict(modelo, siniestro))

    # TODO 4.14: escribir el CSV de salida (con las columnas de OUTPUT_COLUMNS)
    #   en la ruta --output, creando la carpeta si no existe. Solo se escribe
    #   cuando todo lo anterior ha ido bien: si algo falla, no debe quedar un
    #   fichero a medias.
    carpeta = os.path.dirname(args.output)  ##esta parte no tenia ni idea me la ha codificado claude
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        escritor.writeheader()
        for prediccion in predicciones:
            escritor.writerow(prediccion.model_dump())

    # TODO 4.15: mostrar cuantos siniestros se han predicho y terminar con
    #   codigo de salida 0.
    print(f"{len(predicciones)} siniestros predichos -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
