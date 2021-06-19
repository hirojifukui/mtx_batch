function signup_validation()
{
var uemail = document.registration.email;
var passid = document.registration.passid;
var umobile = document.registration.mobile;
var uadd = document.registration.address;
var ucountry = document.registration.country;
var uzip = document.registration.zip;

var umsex = document.registration.msex;
var ufsex = document.registration.fsex; 
if(ValidateEmail(uemail))
{
  if(passid_validation(passid,7,12))
  {
    if(mobile_validation(umobile))
    {
    }
  } 
}
return false;
}

function ValidateEmail(uemail)
{
var mailformat = /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/;
if(uemail.value.match(mailformat))
{
return true;
}
else
{
alert("You have entered an invalid email address!");
uemail.focus();
return false;
}
}
function passid_validation(passid,mx,my)
{
var passid_len = passid.value.length;
if (passid_len == 0 ||passid_len >= my || passid_len < mx)
{
alert("Password should not be empty / length be between "+mx+" to "+my);
passid.focus();
return false;
}
return true;
}
  function mobile_validation(umobile)
  {
  var phoneno = /^\d{10}$/;
  if((umobile.value.match(phoneno))
        {
      return true;
        }
      else
        {
        alert("message");
        umobile.focus();
        return false;
        }
  }

// After form loads focus will go to User id field.
  function firstfocus()
  {
  var email = document.sign_up.email.focus();
  return true;
  }
// This function will validate Email.
  function ValidateEmail()
  {
  var uemail = document.sign_up.email;
  var mailformat = /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/;
  if(uemail.value.match(mailformat))
  {
  document.sign_up.desc.focus();
  return true;
  }
  else
  {
  alert("Enter a valid email address.");
  uemail.focus();
  return false;
  }
  }
// Focus goes to next field i.e. Password.
  document.sign_up.password.focus();
  return true;
  }
// This function will validate Password.
  function passid_validation(mi, mx)
  {
  var passid = document.sign_up.password;
  var passid_len = passid.value.length;
  if (passid_len == 0 ||passid_len >= mx || passid_len < mi)
  {
  alert("Password should not be empty / length be between "+mi+" to "+mx);
  passid.focus();
  return false;
  }
// Focus goes to next field.
  document.sign_up.mobile.focus();
  return true;
  }

// Focus goes to next field i.e. Name.
  document.registration.username.focus();
  return true;
  }
// This function will validate Name.
  function allLetter()
  { 
  var uname = document.registration.username;
  var letters = /^[A-Za-z]+$/;
  if(uname.value.match(letters))
  {
  // Focus goes to next field i.e. Address.
  document.registration.address.focus();
  return true;
  }
  else
  {
  alert('Username must have alphabet characters only');
  uname.focus();

  return false;
  }
  }
  // This function will validate Address.
  function alphanumeric()
  { 
  var uadd = document.registration.address;
  var letters = /^[0-9a-zA-Z]+$/;
  if(uadd.value.match(letters))
  {
  // Focus goes to next field i.e. Country.
  document.registration.country.focus();
  return true;
  }
  else
  {
  alert('User address must have alphanumeric characters only');
  uadd.focus();
  return false;
  }
  }