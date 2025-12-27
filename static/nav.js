// static/nav.js — Router muy simple por hash para alternar vistas
document.addEventListener('DOMContentLoaded', () => {
  const routes = {
    '/enhance': 'view-enhance',
    '/vectorize': 'view-vectorize',
    '/halftone': 'view-halftone',
    '/bgremove': 'view-bgremove',
    '/more': 'view-more',
  };

  const tabs = Array.from(document.querySelectorAll('.tabs .tab'));

  function setActive(viewId){
    // Mostrar/ocultar vistas
    document.querySelectorAll('.view').forEach(v => {
      v.hidden = (v.id !== viewId);
    });

    // Activar pestaña
    tabs.forEach(a => {
      const target = a.getAttribute('data-view');
      if (!target) return;
      const isActive = ('view-' + target) === viewId;
      a.classList.toggle('active', isActive);
      a.setAttribute('aria-current', isActive ? 'page' : 'false');
    });

    // Scroll al inicio del área de trabajo
    const ws = document.querySelector('.workspace');
    ws?.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function resolve(){
    const hash = location.hash || '#/enhance';
    const path = hash.replace('#', '');
    const viewId = routes[path] || routes['/enhance'];
    setActive(viewId);
  }

  window.addEventListener('hashchange', resolve);
  resolve(); // inicial
});
