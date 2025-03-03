import React, { useState } from "react";

export default function App() {
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [smoker, setSmoker] = useState(false);
  const [plotUrl, setPlotUrl] = useState(null); // Stocke l'URL du graphique
  const [fileContent, setFileContent] = useState(null); // Stocke le contenu du fichier texte

  const handleSubmit = async (e) => {
    e.preventDefault();

    const patientData = { name, age, smoker };

    try {
      const response = await fetch("http://localhost:5000/run-python", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patientData),
      });

      const data = await response.json();
      alert(data.result);

      // Récupérer et afficher le graphique
      fetchPlot();

    } catch (error) {
      console.error("Erreur:", error);
      alert("Erreur lors de l'exécution du script Python.");
    }
  };

  // Fonction pour récupérer le graphique
  const fetchPlot = async () => {
    try {
      const response = await fetch("http://localhost:5000/generate-plot");
      if (!response.ok) throw new Error("Erreur lors de la récupération du graphique.");
      
      setPlotUrl(response.url);
    } catch (error) {
      console.error("Erreur lors du chargement du graphique:", error);
      alert("Impossible de générer le graphique.");
    }
  };

  // Fonction pour réinitialiser l'affichage
  const resetForm = () => {
    setPlotUrl(null);
    setFileContent(null);
    setName("");
    setAge("");
    setSmoker(false);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center space-y-6">
      
      {/* Affichage du formulaire uniquement si le graphique n'est pas encore généré */}
      {!plotUrl ? (
        <div className="bg-white p-8 rounded-lg shadow-lg w-full max-w-md">
          <h1 className="text-3xl font-semibold text-center text-gray-800 mb-6">
            Formulaire Patient
          </h1>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Champ Nom */}
            <div className="flex flex-col">
              <label htmlFor="name" className="block text-sm font-medium text-gray-700">Nom</label>
              <input
                type="text"
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1 block w-full px-4 py-2 border rounded-md shadow-sm focus:ring-2 focus:ring-teal-400"
                required
              />
            </div>

            {/* Champ Âge */}
            <div className="flex flex-col">
              <label htmlFor="age" className="block text-sm font-medium text-gray-700">Âge</label>
              <input
                type="number"
                id="age"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                className="mt-1 block w-full px-4 py-2 border rounded-md shadow-sm focus:ring-2 focus:ring-teal-400"
                required
              />
            </div>

            {/* Champ Fumeur */}
            <div className="flex flex-col">
              <label className="block text-sm font-medium text-gray-700">Fumeur ?</label>
              <div className="flex items-center space-x-4">
                <label className="inline-flex items-center">
                  <input
                    type="radio"
                    name="smoker"
                    checked={smoker === true}
                    onChange={() => setSmoker(true)}
                    className="form-radio h-5 w-5 text-teal-600"
                  />
                  <span className="ml-2 text-gray-700">Oui</span>
                </label>
                <label className="inline-flex items-center">
                  <input
                    type="radio"
                    name="smoker"
                    checked={smoker === false}
                    onChange={() => setSmoker(false)}
                    className="form-radio h-5 w-5 text-teal-600"
                  />
                  <span className="ml-2 text-gray-700">Non</span>
                </label>
              </div>
            </div>

            {/* Bouton de soumission */}
            <button
              type="submit"
              className="w-full bg-teal-600 text-white py-2 rounded-md hover:bg-teal-700 focus:ring-2 focus:ring-teal-400"
            >
              Soumettre
            </button>
          </form>
        </div>
      ) :
      (
        // Affichage du graphique et du fichier texte
        <div className="flex flex-col items-center space-y-4 mt-6">
          <h1 className="text-2xl font-semibold">Résultats</h1>
          
          {/* Graphique */}
          <img src={plotUrl} alt="Graphique Matplotlib" className="mt-4 border rounded-lg shadow-md"/>
          
          {/* Contenu du fichier texte */}
          {fileContent && (
            <div className="bg-gray-100 p-4 rounded-lg shadow-md max-w-md">
              <h2 className="text-lg font-semibold">Détails du Patient :</h2>
              <pre className="text-gray-700 whitespace-pre-wrap">{fileContent}</pre>
            </div>
          )}

          {/* Bouton pour recommencer */}
          <button
            onClick={resetForm}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
          >
            Recommencer
          </button>
        </div>
      )}

    </div>
  );
}
