set nocompatible

filetype plugin indent on

syntax enable

" use vim-plug because of its minimalistic
"¯\_(ツ)_/¯

" auto source vimrc
autocmd! BufWritePost $MYVIMRC source $MYVIMRC | echom "Reloaded $MYVIMRC"

let mapleader=','
let maplocalleader='+'

set autoindent
set backspace=indent,eol,start
set complete-=i
set smarttab

set nrformats-=octal

set incsearch
set hlsearch
set noshowcmd
set notitle
set nowrap
set showmode
set hidden
set clipboard=unnamed
set mouse=a
set conceallevel=0
set foldmethod=indent "indent-syntax-manual
set wildmode=longest:full,full
set wildignore+=*.pyc,*.o,*.obj,*.swp,*.class,*.DS_Store,*.min.*
set wildchar=<Tab>
set shortmess+=Ic
set pastetoggle=<F2>
set expandtab
set shiftwidth=4
set tabstop=4
set softtabstop=4
set nojoinspaces
set splitright
set splitbelow

" Performance tuning
set lazyredraw
set nocursorline
set ignorecase
set smartcase

" Misc
set nobackup
set noswapfile
set nowritebackup
set undofile
set undodir=~/.config/vim/undo
set undolevels=9999

if &encoding ==# 'latin1' && has('gui_running')
    set encoding=utf-8
endif

set list
let &listchars='tab:\uBB\uBB,trail:\uB7,nbsp:~'
if &listchars ==# 'eol:$'
    set listchars=tab:>\ ,trail:-,extends:>,precedes:<,nbsp:+
endif

if v:version > 703 || v:version == 703 && has("patch541")
    set formatoptions+=j " Delete comment character when joining commented lines
endif

if has('path_extra')
    setglobal tags-=./tags tags-=./tags; tags^=./tags;
endif

if &shell =~# 'fish$' && (v:version < 704 || v:version == 704 && !has('patch276'))
    set shell=/usr/bin/env\ bash
endif

set autoread

if &history < 1000
    set history=1000
endif
if &tabpagemax < 50
    set tabpagemax=50
endif
if !empty(&viminfo)
    set viminfo^=!
endif
set sessionoptions-=options
set viewoptions-=options

" Allow color schemes to do bright colors without forcing bold.
if &t_Co == 8 && $TERM !~# '^Eterm'
    set t_Co=16
endif

if v:version < 700 || exists('loaded_bclose') || &cp
  finish
endif
let loaded_bclose = 1
if !exists('bclose_multiple')
  let bclose_multiple = 1
endif

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Mapping
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Mapping
nnoremap ; :
imap jj <Esc>

" manual folding
inoremap <F5> <C-O>za
nnoremap <F5> za
onoremap <F5> <C-C>za
vnoremap <F5> zf
" Toggle show/hide invisible chars
nnoremap <leader>I :set list!<cr>
nnoremap \\ :let @/=''<CR>:noh<CR>
nnoremap <silent> <leader>p :%s///g<CR>
nnoremap <silent> <leader>i gg=G<CR>
nnoremap <leader># :g/\v^(#\|$)/d_<CR>
nnoremap <leader>b :ls<CR>:buffer<space>
nnoremap <leader>d :w !diff % -<CR>
nnoremap <leader>S :so $MYVIMRC<CR>
nnoremap <leader>l :set list! list?<CR>
nnoremap <leader>t :%s/\s\+$//e<CR>
" Remove the Windows ^M - when the encodings gets messed up
" for somereason bufread doesn't catch  it first
nnoremap <leader>m mmHmt:%s/<C-V><cr>//ge<cr>'tzt'm nnoremap <leader>w :set wrap! wrap?<CR>
" When you press <leader>r you can search and replace the selected text
nnoremap <leader>ml :call AppendModeLine()<CR>
" Visual mode pressing * or # searches for the current selection
" Super useful! From an idea by Michael Naumann
vnoremap <silent> * :<C-u>call VisualSelection('', '')<CR>/<C-R>=@/<CR><CR>
vnoremap <silent> # :<C-u>call VisualSelection('', '')<CR>?<C-R>=@/<CR><CR>
" this is for transient
" Smart way to move between windows
map <C-j> <C-W>j
map <C-k> <C-W>k
map <C-h> <C-W>h
map <C-l> <C-W>l
" Opens a new tab with the current buffer's path
" Super useful when editing files in the same directory
nnoremap <leader>te :tabedit <C-r>=expand("%:p:h")<cr>/
nnoremap <leader>vs :vsplit <C-r>=expand("%:p:h")<cr>/
nnoremap <leader>s :split <C-r>=expand("%:p:h")<cr>/
nnoremap <silent><nowait> <space>l :bNext<cr>
nnoremap <silent><nowait> <space>h :bprevious<cr>
vmap J :m '>+1<cr>gv=gv<CR>
vmap K :m '<-2<cr>gv=gv<CR>

" remove Ex mode
map Q <Nop>
" added yank to clipboard shortcut
noremap <M-Y> "*y
noremap <M-P> "*p
noremap <M-y> "+y
noremap <M-p> "+p
" use this when lightline is not in use for minimal
nnoremap <F2> :set invpaste paste?<CR>
imap <F2><C-O>:set invpaste paste?<CR>
" quick resize for split
nnoremap <silent> <leader>+ :exe "resize " . (winheight(0) * 3/2)<CR>
nnoremap <silent> <leader>- :exe "resize " . (winheight(0) * 2/3)<CR>

" Append modeline after last line in buffer.
" Use substitute() instead of printf() to handle '%%s' modeline in LaTeX
" files.
function! AppendModeLine()
    let l:modeline = printf("vim: set ft=%s ts=%d sw=%d tw=%d %set :",
                \ &filetype, &tabstop, &shiftwidth, &textwidth, &expandtab ? '' : 'no')
    let l:modeline = substitute(&commentstring, "%s", l:modeline, "")
    call append(line("$"), l:modeline)
endfunction

function! CmdLine(str)
    call feedkeys(":" . a:str)
endfunction

function! VisualSelection(direction, extra_filter) range
    let l:saved_reg = @"
    execute "normal! vgvy"

    let l:pattern = escape(@", "\\/.*'$^~[]")
    let l:pattern = substitute(l:pattern, "\n$", "", "")

    if a:direction == 'gv'
        call CmdLine("Ack '" . l:pattern . "' " )
    elseif a:direction == 'replace'
        call CmdLine("%s" . '/'. l:pattern . '/')
    endif

    let @/ = l:pattern
    let @" = l:saved_reg
endfunction