from fastapi import FastAPI, HTTPException
import requests
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()

#Routet die HTTP GET Requests zu dem bestimmten Pfad
@app.get("/api/V1/recommend/")
#getRecommandation ist die Funktion die mithilfe dem einzugebenen Standort die API Abruft und
# zusätzlich die Bekleidungsempfehlung in Form einer JSON ausgibt.
def getRecommandation(lat: float, long: float):
    """
    :param lat: Latitude des Nutzers
    :param long: Longitutde des Nutzers
    """
    #Der API Key wird als Secret aus der Datei ".env" geladen und ist somit versteckt.
    apikey = os.getenv("API_KEY")
    APIHostname = os.getenv("APIHostname")
    url = f"{APIHostname}?lat={lat}&lon={long}&exclude=alerts,daily,minutely&units=metric&appid={apikey}"
    #Der API Call wird ausgeführt. Zusätzlich ist ein timeout ab einer Sekunde eingestellt.
    r = requests.get(url, timeout=1)

    #Folgend werden Fehlermeldungen von der API verarbeitet
    if r.status_code == 401:
        print("Es wurde ein nicht vorhandener oder falscher API-Key verwendet.")
        raise HTTPException(status_code=401, detail="„Invalid API key")
    elif r.status_code == 408:
        print("Der API Service kann aktuell nicht erreicht werden.")
        raise HTTPException(status_code=408, detail="„API service can't be reached")
    #Die API Daten werden in das Dicionary "weather_dict" gespeichert.
    weather_dict = r.json()

    # Extrahierung der Temperatur aus der JSON
    temp = weather_dict["hourly"][0]["temp"]

    # Extrahierung des UV-Index aus der JSON
    uvi = weather_dict["hourly"][0]["uvi"]

    # Extrahierung der Regenwahrscheinlichkeit aus der JSON
    pop = weather_dict["hourly"][0]["pop"]
    # Die Funktion rateVaribles wird mit den ermittelten Variablen aufgerufen.
    clothes, risk, umbrella = rateVaribles(temp, uvi, pop)
    # Die Funktion "createAndReturnJson" wird aufgerufen und die JSON wird wiedergegeben.
    return createAndReturnJson(clothes, risk, umbrella)


def rateVaribles(temp, uvi, pop):
    """
    :param temp: Beinhaltet die aktuelle Temperatur
    :param uvi: Beinhaltet den aktuellen UV-Index
    :param pop: Beinhaltet die aktuelle Regenwahrscheinlichkeit
    :return:
    """
    # Je nach Höhe der Temperatur enthält die Variable clothes den String "coat", "sweater" oder "thsirt".
    if temp <= 5:
        clothes = "coat"
    elif temp > 5 and temp <= 12:
        clothes = "sweater"
    elif temp > 12:
        clothes = "tshirt"

    # Je nach Höhe des UV-Index enthält die Variable risk den String "coat", "sweater" oder "thsirt".
    if uvi <= 2:
        risk = "low"
    elif uvi > 2 and uvi < 6:
        risk = "moderate"
    elif uvi >= 6:
        risk = "high"

    # setUmbrella
    if pop < 0.1:
        umbrella = "no"
    elif pop >= 0.1:
        umbrella = "yes"
    return clothes, risk, umbrella


def createAndReturnJson(clothes, risk, umbrella):
    """
    :param clothes: beinhaltet die aktuelle Bekleidungsempfehlung
    :param risk: gibt den Nutzer Auskunft über die aktuelle Schädlichkeit der UV-Strahlung
    :param umbrella: teilt dem Nutzer mit, ob dieser aktuell einen Regenschirm benötigt oder nicht
    :return:
    """
    response = {
        "clothes": clothes,
        "risk": risk,
        "umbrella": umbrella
    }
    return response

"""
Überreste aus der Testzeit:
def getWeatherInformation(lat, long, apikey):
    temp, uvi, pop = getRecommandation(lat, long, apikey)
    clothes, risk, umbrella = rateVaribles(temp, uvi, pop)
    createAndSendJson(clothes, risk, umbrella)
"""


