<?php
namespace docker {
	function adminer_object() {
		require_once('plugins/plugin.php');

		class Adminer extends \AdminerPlugin {
			function credentials() {
				return [
					$_ENV['ADMINER_DEFAULT_SERVER'] ?: 'db',
					$_ENV['ADMINER_DEFAULT_USERNAME'] ?: 'postgres',
					$_ENV['ADMINER_DEFAULT_PASSWORD'] ?? '',
				];
			}

			function login($login, $password) {
				return true;
			}

			function _callParent($function, $args) {
				if ($function === 'loginForm') {
					ob_start();
					$return = \Adminer::loginForm();
					$form = ob_get_clean();

					$form = str_replace('name="auth[server]" value="" title="hostname[:port]"', 'name="auth[server]" value="'.htmlspecialchars($_ENV['ADMINER_DEFAULT_SERVER'] ?: 'db').'" title="hostname[:port]"', $form);
					$form = str_replace('name="auth[username]" id="username" value=""', 'name="auth[username]" id="username" value="'.htmlspecialchars($_ENV['ADMINER_DEFAULT_USERNAME'] ?: 'postgres').'"', $form);
					$form = str_replace('name="auth[db]" value=""', 'name="auth[db]" value="'.htmlspecialchars($_ENV['ADMINER_DEFAULT_DATABASE'] ?: 'postgres').'"', $form);

					echo $form;
					return $return;
				}

				return parent::_callParent($function, $args);
			}
		}

		$plugins = [];
		foreach (glob('plugins-enabled/*.php') as $plugin) {
			$plugins[] = require($plugin);
		}

		return new Adminer($plugins);
	}
}

namespace {
	if (basename($_SERVER['DOCUMENT_URI'] ?? $_SERVER['REQUEST_URI']) === 'adminer.css' && is_readable('adminer.css')) {
		header('Content-Type: text/css');
		readfile('adminer.css');
		exit;
	}

	function adminer_object() {
		return \docker\adminer_object();
	}

	if (($_SERVER['QUERY_STRING'] ?? '') === '' || empty($_COOKIE['adminer_permanent'])) {
		$_POST['auth'] = [
			'driver' => $_ENV['ADMINER_DEFAULT_DRIVER'] ?: 'pgsql',
			'server' => $_ENV['ADMINER_DEFAULT_SERVER'] ?: 'db',
			'username' => $_ENV['ADMINER_DEFAULT_USERNAME'] ?: 'postgres',
			'password' => $_ENV['ADMINER_DEFAULT_PASSWORD'] ?? '',
			'db' => $_ENV['ADMINER_DEFAULT_DATABASE'] ?: 'postgres',
			'permanent' => 1,
		];
	}

	require('adminer.php');
}
