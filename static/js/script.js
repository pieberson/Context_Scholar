document.addEventListener("DOMContentLoaded", () => {
  // === LOADING OVERLAY HANDLER ===
  const overlay = document.getElementById("loadingOverlay");
  console.debug("🔍 DEBUG: Loading overlay element:", overlay);
  const forms = document.querySelectorAll("#normalForm, #experimentForm, #normalFormResult, #experimentFormResult");
  console.debug("🔍 DEBUG: Found forms:", forms.length);
  forms.forEach(form => {
      if (!form) return;
      form.addEventListener("submit", () => {
          console.debug("🔍 DEBUG: Form submitted, showing overlay");
          if (overlay) overlay.style.display = "block";
      });
  });

  // === DATE RANGE TOGGLE (for results page) ===
  const customRangeRadio = document.getElementById("customRange");
  const mostRecentRadio = document.getElementById("mostRecent");
  const dateRangeFields = document.getElementById("customYearFields");

  if (customRangeRadio && mostRecentRadio && dateRangeFields) {
    function toggleDateRange() {
      dateRangeFields.style.display = customRangeRadio.checked ? "flex" : "none";
    }

    toggleDateRange();
    customRangeRadio.addEventListener("change", toggleDateRange);
    mostRecentRadio.addEventListener("change", toggleDateRange);
  } else {
    console.debug("🧠 DEBUG: Date range controls not found — skipping date toggle setup.");
  }

  // === EXPERIMENT TOGGLE (for index page) ===
  const experimentToggle = document.getElementById("experimentToggle");
  const experimentForm = document.getElementById("experimentForm");
  const normalForm = document.getElementById("normalForm");

  if (experimentToggle && experimentForm && normalForm) {
    function updateForms() {
      if (experimentToggle.checked) {
        experimentForm.style.display = "block";
        normalForm.style.display = "none";
      } else {
        experimentForm.style.display = "none";
        normalForm.style.display = "block";
      }
    }

    experimentToggle.addEventListener("change", updateForms);
    updateForms(); // initialize display on load
  } else {
    console.debug("🧠 DEBUG: Experiment toggle or base forms not found — skipping updateForms setup.");
  } 

  // for bookmark removal 
  document.querySelectorAll('.remove-bookmark-btn, .remove-save-btn').forEach(button => {
        button.addEventListener('click', async () => {
            const paperId = button.dataset.id;

            const response = await fetch('/remove-bookmark', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ id: paperId })
            });

            const result = await response.json();

            if (result.status === 'success') {
                // Optional: show a confirmation modal
                const removeModalEl = document.getElementById('removeModal');
                if (removeModalEl) {
                    const removeModal = new bootstrap.Modal(removeModalEl);
                    removeModal.show();
                }

                // Dynamically remove the paper element from the DOM
                const paperItem = button.closest('.paper-item');
                const paperCard = button.closest('.paper-card');
                
                if (paperItem) {
                    paperItem.classList.add('fade-out'); // optional animation
                    setTimeout(() => paperItem.remove(), 300);
                } 

                if (paperCard) {
                        paperCard.classList.add('fade-out');
                        setTimeout(() => paperCard.remove(), 300);
                    }

            } else {
                alert(result.message || 'Something went wrong.');
            }
        });
    });

    // for save paper
    document.querySelectorAll('.save-btn').forEach(button => {
    button.addEventListener('click', async () => {
      const paperData = {
        title: button.dataset.title,
        authors: button.dataset.authors,
        year: button.dataset.year,
        url: button.dataset.url,
        citations: button.dataset.citations,
        score: button.dataset.score
      };

      try {
        const response = await fetch('/save-paper', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(paperData)
        });

        const result = await response.json();

        if (result.status === 'success') {
          const saveModal = new bootstrap.Modal(document.getElementById('saveModal'));
          saveModal.show();
        }
        else if (result.status === 'exists') {
            const alreadySavedModal = new bootstrap.Modal(document.getElementById('alreadySavedModal'));
            alreadySavedModal.show();
        } 
        else {
          alert(result.message);
        }
      } catch (err) {
        alert("An error occurred. Please try again.");
      }
    });
  });
}); 

