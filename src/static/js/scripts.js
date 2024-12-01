document.getElementById("predict").addEventListener("click", async () => {
    const fileInput = document.getElementById("file-upload");
    const file = fileInput.files[0];
  
    if (!file) {
      alert("Please upload an image before predicting!");
      return;
    }
  
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      const response = await fetch("/predict", {  // FastAPI endpoint
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) {
        throw new Error(await response.text());
      }
  
      const result = await response.json();
      alert(`Prediction: ${result.animal_type}`);
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to get prediction. Please try again.");
    }
  });
  