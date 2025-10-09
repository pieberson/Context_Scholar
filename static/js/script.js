document.addEventListener("DOMContentLoaded", () => {
  // === LOADING OVERLAY HANDLER ===
  const overlay = document.getElementById("loadingOverlay");
  document.querySelectorAll("#normalForm, #experimentForm, #normalFormResult, #experimentFormResult")
    .forEach(form => {
      if (!form) return;
      form.addEventListener("submit", () => {
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
});
