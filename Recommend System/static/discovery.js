// Arrows move one visible row. Touchpads, touch swipes, and keyboard also work.
document.querySelectorAll('.movie-rail').forEach(rail => {
  const buttons = [...document.querySelectorAll(`[data-rail="${rail.id}"]`)];
  const refresh = () => buttons.forEach(button => {
    button.disabled = Number(button.dataset.direction) < 0
      ? rail.scrollLeft <= 4
      : rail.scrollLeft + rail.clientWidth >= rail.scrollWidth - 2;
  });
  buttons.forEach(button => button.addEventListener('click', () => {
    rail.scrollBy({left: Number(button.dataset.direction) * rail.clientWidth * .85,
      behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});
  }));
  rail.addEventListener('scroll', refresh, {passive:true});
  rail.addEventListener('keydown', event => {
    if (event.target === rail && ['ArrowLeft','ArrowRight'].includes(event.key)) {
      event.preventDefault(); rail.scrollBy({left:event.key === 'ArrowLeft' ? -200 : 200});
    }
  });
  new ResizeObserver(refresh).observe(rail);
  refresh();
});
