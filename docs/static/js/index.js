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

  // Same treatment for strips that are not inside a tabbed viewer (zero-shot transfer)
  var stripObserver = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      entry.isIntersecting ? playPanel(entry.target) : pausePanel(entry.target);
    });
  }, { threshold: 0.25 });
  document.querySelectorAll('.hw-zs .video-strip').forEach(function(v) { stripObserver.observe(v); });

  // File viewer: fetch the file the first time the disclosure is opened.
  // Covers the simulation policy-code blocks and every artifact in the hardware trace.
  document.querySelectorAll('details.policy-code[data-src], details.file-view[data-src]').forEach(function(details) {
    details.addEventListener('toggle', function() {
      if (!details.open || details.dataset.loaded) return;
      details.dataset.loaded = 'true';
      var code = details.querySelector('code');
      var lang = details.dataset.lang || 'python';
      fetch(details.dataset.src)
        .then(function(r) {
          if (!r.ok) throw new Error(r.status);
          return r.text();
        })
        .then(function(text) {
          code.textContent = text;
          // hljs has no diff/plain grammar loaded here; only highlight what it knows.
          if (window.hljs && lang === 'python') {
            code.classList.add('language-python');
            hljs.highlightElement(code);
          }
        })
        .catch(function() {
          code.textContent = 'Could not load ' + details.dataset.src;
          delete details.dataset.loaded;
        });
    });
  });

  // Table of contents: highlight whichever section is currently in view.
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('.toc a'));
  if (tocLinks.length) {
    var targets = tocLinks
      .map(function(a) {
        var el = document.getElementById(a.getAttribute('href').slice(1));
        return el ? { link: a, el: el } : null;
      })
      .filter(Boolean);

    var markCurrent = function() {
      // The entry whose top is the last one above the marker line is the active one.
      var marker = window.innerHeight * 0.3;
      var current = targets[0];
      targets.forEach(function(t) {
        if (t.el.getBoundingClientRect().top <= marker) current = t;
      });
      tocLinks.forEach(function(a) { a.classList.toggle('is-current', current && a === current.link); });
    };

    var ticking = false;
    window.addEventListener('scroll', function() {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(function() { markCurrent(); ticking = false; });
    }, { passive: true });
    markCurrent();
  }
});
