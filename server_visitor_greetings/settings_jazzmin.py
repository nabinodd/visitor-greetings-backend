JAZZMIN_UI_TWEAKS = {
   'theme': 'materia',
}

JAZZMIN_SETTINGS = {
   'site_logo' : 'icon.png',
   'login_logo': 'icon.png',
   'site_icon': 'icon.png',
   
   'site_logo_classes': 'ms-2',

   'welcome_sign': "Greetings Superadmin",   
   'copyright': "Greetings",

   'topmenu_links': [
      {'name': 'Home',  'url': '/admin',},
      {'name': 'Support', 'url': 'https://techcolab.org', 'new_window': True},
   ],

   'usermenu_links': [
      {'name': 'Support', 'url': 'https://techcolab.org', 'new_window': True},
      {'model': 'auth.user'}
   ],
   
   'related_modal_active': True,
   'use_google_fonts_cdn': True,
   # 'show_ui_builder': True,

}