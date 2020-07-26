const current = document.querySelector("#current");
const imgs = document.querySelectorAll(".imgs img");
const opacity = 0.75;

// Set first img opacity
imgs[0].style.opacity = opacity;

imgs.forEach(img => img.addEventListener("click", imgClick));
sessionStorage.setItem('current-img', current.src)
function imgClick(e) {
  // Reset the opacity
  imgs.forEach(img => (img.style.opacity = 1));

  // Change current image to src of clicked image
  current.src = e.target.src;

  // Add fade in class
  current.classList.add("fade-in");

  // Remove fade-in class after .5 seconds
  setTimeout(() => current.classList.remove("fade-in"), 500);

  // Change the opacity to opacity var
  e.target.style.opacity = opacity;

  sessionStorage.setItem('current-img', current.src);

  // store current image id in hidden field to be used @ form.request
  document.getElementById('current-image-src').value = current.src;

  // reset question input as new iamge has been chosen
  document.getElementById('question').value = '';
  document.getElementById('question').placeholder = 'Please enter your question here';

  // delete the answer list from the previous question
  var element = document.getElementById('predicted-answers');
  element.parentNode.removeChild(element);

  // scroll to top of page
  window.scrollTo({
    top: 0,
    left: 0,
    behavior: 'smooth'
  });
}