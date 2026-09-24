document.addEventListener('DOMContentLoaded', function() {
  // Only the visible task panel plays; hidden panels are paused.
  function playPanel(panel) {
    panel.querySelectorAll('video').forEach(function(v) {
      v.preload = 'auto';
      v.play().catch(function() {});
    });
  }
  function pausePanel(panel) {
    panel.querySelectorAll('video').forEach(function(v) { v.pause(); });
  }

  // Task tabs
  document.querySelectorAll('.task-viewer').forEach(function(viewer) {
    var tabs = viewer.querySelectorAll('.tabs li');
    var panels = viewer.querySelectorAll('.task-panel');
    tabs.forEach(function(tab) {
      tab.addEventListener('click', function() {
        tabs.forEach(function(t) { t.classList.toggle('is-active', t === tab); });
        panels.forEach(function(p) {
          var active = p.dataset.task === tab.dataset.task;
          p.classList.toggle('is-active', active);
          active ? playPanel(p) : pausePanel(p);
        });
      });
    });
  });

  // Autoplay the active panels once they scroll into view
  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      var panel = entry.target.querySelector('.task-panel.is-active');
      if (!panel) return;
      entry.isIntersecting ? playPanel(panel) : pausePanel(panel);
    });
  }, { threshold: 0.25 });
  document.querySelectorAll('.task-viewer').forEach(function(v) { observer.observe(v); });

  // Policy code viewer: fetch the code the first time it is opened
  document.querySelectorAll('details.policy-code').forEach(function(details) {
    details.addEventListener('toggle', function() {
      if (!details.open || details.dataset.loaded) return;
      details.dataset.loaded = 'true';
      var code = details.querySelector('code');
      fetch(details.dataset.src)
        .then(function(r) {
          if (!r.ok) throw new Error(r.status);
          return r.text();
        })
        .then(function(text) {
          code.textContent = text;
          if (window.hljs) hljs.highlightElement(code);
        })
        .catch(function() {
          code.textContent = 'Could not load policy code.';
          delete details.dataset.loaded;
        });
    });
  });
});
